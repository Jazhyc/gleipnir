"""Restore a checksum-bound serving environment onto fresh project paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_bundle(bundle: Path) -> dict:
    """Verify all archive/metadata bytes before restoring any environment."""
    manifest = json.loads((bundle / "bundle.json").read_text())
    if manifest.get("schema") != 1:
        raise ValueError("unsupported bundle schema")
    for name, expected in manifest["files"].items():
        if Path(name).name != name:
            raise ValueError("bundle files must be direct children")
        path = bundle / name
        if (
            path.stat().st_size != expected["bytes"]
            or digest(path) != expected["sha256"]
        ):
            raise ValueError(f"bundle checksum mismatch: {name}")
    required = {"runtime", "caches", "source", "context"}
    if set(manifest["archives"]) != required:
        raise ValueError("bundle must include runtime, caches, source and context")
    references = {*manifest["archives"].values(), *manifest["metadata"].values()}
    if not references <= manifest["files"].keys():
        raise ValueError("unbound bundle archive or metadata")
    return manifest


def extract_archive(archive: Path, target: Path) -> None:
    """Extract trusted, verified bytes with traversal and link protections."""
    if archive.name.endswith(".zst"):
        with subprocess.Popen(
            ["zstd", "-dc", str(archive)], stdout=subprocess.PIPE
        ) as process:
            try:
                with tarfile.open(fileobj=process.stdout, mode="r|") as contents:
                    contents.extractall(target, filter="data")
            finally:
                if process.stdout is not None:
                    process.stdout.close()
            if process.wait() != 0:
                raise ValueError(f"decompression failed: {archive.name}")
    else:
        with tarfile.open(archive, "r:gz") as contents:
            contents.extractall(target, filter="data")


def dependency_path(relative: str, root: Path, runtime: Path) -> Path:
    """Map only the dependency trees bound by the ordinary serving launcher."""
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("unsafe dependency binding path")
    prefixes = {
        ".venv/": "venv",
        ".cache/kernels/triton/": "triton",
        ".cache/kernels/nvidia_mxfp8/": "mxfp8",
        ".cache/kernels/fa4/": "fa4",
    }
    for prefix, name in prefixes.items():
        if relative.startswith(prefix):
            return runtime / name / relative.removeprefix(prefix)
    if relative not in {"uv.lock", "pyproject.toml"}:
        raise ValueError(f"unsupported dependency binding: {relative}")
    return root / relative


def restore_bundle(bundle: Path, root: Path, runtime: Path, tokenizer: Path) -> dict:
    """Restore fresh destinations; never overwrite a resident worker or project."""
    manifest = verify_bundle(bundle)
    destinations = [p.absolute() for p in [root, runtime, tokenizer]]
    root, runtime, tokenizer = destinations
    for index, path in enumerate(destinations):
        if path.exists() or path.is_symlink():
            raise ValueError(f"restore destination already exists: {path}")
        for other in destinations[index + 1 :]:
            if (
                path == other
                or path.is_relative_to(other)
                or other.is_relative_to(path)
            ):
                raise ValueError("restore destinations must be separate")
    metadata = manifest["metadata"]
    receipt = json.loads((bundle / metadata["runtime_manifest"]).read_text())
    source_files = json.loads((bundle / metadata["binding_source_files"]).read_text())
    original_root = receipt["binding"]["source_root"]
    with tempfile.TemporaryDirectory(prefix="gleipnir-bundle-") as temporary:
        staging = Path(temporary)
        project, packages = staging / "project", staging / "packages"
        project.mkdir()
        packages.mkdir()
        archives = manifest["archives"]
        extract_archive(bundle / archives["runtime"], packages)
        for kind in ["source", "context", "caches"]:
            extract_archive(bundle / archives[kind], project)
        staged_runtime = packages / "gleipnir-serving-runtime"
        staged_tokenizer = packages / "gleipnir-gigatoken-0.10.0"
        for relative, expected in receipt["binding"]["files"].items():
            path = dependency_path(relative, project, staged_runtime)
            if not path.exists() and ".git" in Path(relative).parts:
                # Archives omit Git objects; preserve the bound source marker.
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(source_files[relative])
            if digest(path) != expected:
                raise ValueError(f"restored dependency mismatch: {relative}")
        if (
            not (staged_runtime / "venv/bin/python").is_file()
            or not staged_tokenizer.is_dir()
        ):
            raise ValueError("bundle interpreter or native tokenizer is missing")
        for relative, name in {
            ".venv": "venv",
            ".cache/kernels/triton": "triton",
            ".cache/kernels/nvidia_mxfp8": "mxfp8",
            ".cache/kernels/fa4": "fa4",
        }.items():
            link = project / relative
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.exists() or link.is_symlink():
                raise ValueError(
                    f"source archive contains dependency destination: {relative}"
                )
            link.symlink_to(runtime / name, target_is_directory=True)
        receipt["binding"]["source_root"] = str(root)
        old_runtime = str(Path(receipt["python"]).parents[2])
        receipt["python"] = str(runtime / "venv/bin/python")
        receipt["path_mappings"] = {
            (
                str(root) + key[len(original_root) :]
                if key.startswith(original_root + "/")
                else key
            ): str(runtime) + value[len(old_runtime) :]
            for key, value in receipt["path_mappings"].items()
        }
        (staged_runtime / "manifest.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
        (staged_runtime / ".gleipnir-runtime.json").write_text(
            json.dumps({"source_root": str(root)}) + "\n"
        )
        for source, destination in [
            (staged_runtime, runtime),
            (staged_tokenizer, tokenizer),
            (project, root),
        ]:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() or destination.is_symlink():
                raise ValueError(
                    f"restore destination appeared during extraction: {destination}"
                )
            shutil.move(str(source), str(destination))
    result = {
        "status": "restored",
        "bundle_sha256": digest(bundle / "bundle.json"),
        "root": str(root),
        "runtime": str(runtime),
        "tokenizer": str(tokenizer),
        "python": receipt["python"],
        "weights_included": False,
    }
    (root / "results/serving_bundle").mkdir(parents=True, exist_ok=True)
    (root / "results/serving_bundle/restore.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--root", type=Path, default=Path("/workspace/gleipnir"))
    parser.add_argument(
        "--runtime", type=Path, default=Path("/tmp/gleipnir-serving-runtime")
    )
    parser.add_argument(
        "--tokenizer", type=Path, default=Path("/tmp/gleipnir-gigatoken-0.10.0")
    )
    args = parser.parse_args()
    if args.verify_only:
        manifest = verify_bundle(args.bundle)
        print(json.dumps({"status": "verified", "files": len(manifest["files"])}))
    else:
        print(
            json.dumps(
                restore_bundle(args.bundle, args.root, args.runtime, args.tokenizer)
            )
        )


if __name__ == "__main__":
    main()
