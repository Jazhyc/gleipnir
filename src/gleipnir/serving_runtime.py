"""Stage immutable serving dependencies locally; keep durable caches shared."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DEFAULT_RUNTIME = Path("/tmp/gleipnir-serving-runtime")


def copy_dependency_tree(source: Path, destination: Path, workers: int = 8) -> None:
    """Copy disjoint package trees concurrently, reusing already copied files."""
    split = {
        ".",
        "torch_compile_cache",
        "torch_compile_cache/torch_aot_compile",
        "lib",
        "lib/python3.12",
        "lib/python3.12/site-packages",
        "lib/python3.12/site-packages/nvidia",
        "lib/python3.12/site-packages/torch",
        "lib/python3.12/site-packages/torch/lib",
        "lib/python3.12/site-packages/transformers",
        "lib/python3.12/site-packages/transformers/models",
    }
    jobs = []
    pending = [(source, destination)]
    while pending:
        src, dst = pending.pop()
        if src.is_dir() and src.relative_to(source).as_posix() in split:
            dst.mkdir(parents=True, exist_ok=True)
            children = list(src.iterdir())
            allowed = {
                c.name
                for c in children
                if c.name != ".git" and not (c.parent == source and c.name == "lib64")
            }
            for obsolete in dst.iterdir():
                if obsolete.name not in allowed:
                    if obsolete.is_dir() and not obsolete.is_symlink():
                        shutil.rmtree(obsolete)
                    else:
                        obsolete.unlink()
            for child in children:
                if child.name != ".git" and not (
                    child.parent == source and child.name == "lib64"
                ):
                    pending.append((child, dst / child.name))
        else:
            jobs.append((src, dst))

    def copy(job):
        src, dst = job
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink() or (dst.exists() and src.is_dir() != dst.is_dir()):
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        suffix = "/" if src.is_dir() else ""
        if suffix:
            dst.mkdir(exist_ok=True)
        subprocess.run(
            [
                "rsync",
                "-aL",
                "--delete",
                "--exclude=.git",
                f"{src}{suffix}",
                f"{dst}{suffix}",
            ],
            check=True,
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(copy, jobs))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_binding(root: Path) -> dict:
    """Bind the lock, interpreter and installed package records without imports."""
    files = [root / "uv.lock", root / "pyproject.toml", root / ".venv/pyvenv.cfg"]
    site = root / ".venv/lib/python3.12/site-packages"
    files.extend(sorted(site.glob("*.dist-info/METADATA")))
    for base in [
        root / ".cache/kernels/triton",
        root / ".cache/kernels/nvidia_mxfp8/frontend",
        root / ".cache/kernels/fa4",
    ]:
        for name in ["pyproject.toml", "setup.py", ".git/HEAD"]:
            if (base / name).is_file():
                files.append(base / name)
    values = {str(p.relative_to(root)): sha(p) for p in files}
    return {"source_root": str(root.resolve()), "files": values}


def stage_runtime(root: Path, target: Path = DEFAULT_RUNTIME) -> dict:
    """Copy packages once per dependency identity, never model/cache artifacts."""
    root, target = root.resolve(), target.resolve()
    if target == root or target.is_relative_to(root) or root.is_relative_to(target):
        raise ValueError("runtime staging must be separate from source")
    binding = runtime_binding(root)
    manifest_path = target / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("binding") == binding and Path(existing["python"]).is_file():
            return existing
    target.mkdir(parents=True, exist_ok=True)
    owner = target / ".gleipnir-runtime.json"
    if owner.exists():
        if json.loads(owner.read_text())["source_root"] != str(root):
            raise ValueError("staging directory belongs to another source")
    elif any(target.iterdir()):
        raise ValueError("refuse to overwrite an unowned staging directory")
    owner.write_text(json.dumps({"source_root": str(root)}) + "\n")
    sources = {
        "venv": root / ".venv",
        "triton": root / ".cache/kernels/triton",
        "mxfp8": root / ".cache/kernels/nvidia_mxfp8",
        "fa4": root / ".cache/kernels/fa4",
    }
    started = time.monotonic()
    mappings = {}
    for name, source in sources.items():
        if not source.is_dir():
            raise ValueError(f"missing runtime dependency: {source}")
        destination = target / name
        destination.mkdir(exist_ok=True)
        copy_dependency_tree(source, destination)
        mappings[str(source)] = str(destination)
    mappings["/tmp/gleipnir-triton-3.7.1"] = str(target / "triton")
    if runtime_binding(root) != binding:
        raise ValueError("dependencies changed during runtime staging")
    # Transfer checksums are checked by rsync; bind package metadata independently.
    for relative, digest in binding["files"].items():
        if relative.startswith(".venv/"):
            copied = target / "venv" / relative.removeprefix(".venv/")
            if sha(copied) != digest:
                raise ValueError(f"staged package identity mismatch: {relative}")
    receipt = {
        "schema": 1,
        "binding": binding,
        "path_mappings": mappings,
        "python": str(target / "venv/bin/python"),
        "staging_seconds": time.monotonic() - started,
        "created_at_unix": time.time(),
        "ephemeral": True,
        "shared_compiler_and_kernel_caches_preserved": True,
    }
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    temporary.replace(manifest_path)
    return receipt


def local_serving_runtime(root: Path, environment: dict[str, str]) -> dict | None:
    """Use a verified staged runtime when present; fail on stale dependencies."""
    target = Path(environment.get("GLEIPNIR_SERVING_RUNTIME", str(DEFAULT_RUNTIME)))
    path = target / "manifest.json"
    if not path.exists():
        return None
    receipt = json.loads(path.read_text())
    if receipt.get("schema") != 1 or receipt["binding"] != runtime_binding(root):
        raise ValueError("staged serving runtime is stale; restage dependencies")
    if not Path(receipt["python"]).is_file():
        raise ValueError("staged serving interpreter is missing")
    mappings = sorted(
        receipt["path_mappings"].items(), key=lambda x: len(x[0]), reverse=True
    )

    def rewrite(value: str) -> str:
        for source, destination in mappings:
            if value == source or value.startswith(source + "/"):
                return destination + value[len(source) :]
        return value

    for key in ["PATH", "PYTHONPATH", "LD_LIBRARY_PATH"]:
        if key in environment:
            environment[key] = os.pathsep.join(
                rewrite(v) for v in environment[key].split(os.pathsep)
            )
    return {"manifest": str(path), "manifest_sha256": sha(path), **receipt}
