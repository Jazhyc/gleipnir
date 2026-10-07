"""Keep durable compiler caches shared and mirror their reads locally."""

import json
import subprocess
import tempfile
import time
from pathlib import Path

from gleipnir.serving_runtime import DEFAULT_RUNTIME, copy_dependency_tree, sha


def inventory(folder: Path) -> dict:
    """Cache entries are immutable by key; detect writes by size and mtime."""
    return {
        str(p.relative_to(folder)): [p.stat().st_size, p.stat().st_mtime_ns]
        for p in folder.rglob("*")
        if p.is_file()
        and not p.is_symlink()
        and not p.name.endswith((".tmp", ".lock"))
        and "locks" not in p.parts
    }


def stage_compiler_mirror(root: Path, target: Path = DEFAULT_RUNTIME) -> dict:
    """Populate one reusable local mirror from the existing shared cache keys."""
    folder = target / "compiler_cache"
    folder.mkdir(parents=True, exist_ok=True)
    mappings = {
        "VLLM_CACHE_ROOT": {
            "shared": str(root / ".cache/vllm/student_injection_awareness_v1"),
            "local": str(folder / "vllm"),
        },
        "TORCHINDUCTOR_CACHE_DIR": {
            "shared": str(root / ".cache/torchinductor/student_injection_awareness_v1"),
            "local": str(folder / "inductor"),
        },
    }
    started = time.monotonic()
    for value in mappings.values():
        copy_dependency_tree(Path(value["shared"]), Path(value["local"]))
    receipt = {
        "source_root": str(root),
        "mappings": mappings,
        "staging_seconds": time.monotonic() - started,
        "new_keys_written_back": True,
    }
    path = folder / "manifest.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def compiler_mirror(root: Path, environment: dict) -> dict | None:
    """Activate a staged mirror without creating a per-experiment cache."""
    if environment.get("GLEIPNIR_COMPILER_MIRROR") != "1":
        return None
    target = Path(environment.get("GLEIPNIR_SERVING_RUNTIME", str(DEFAULT_RUNTIME)))
    path = target / "compiler_cache/manifest.json"
    if not path.exists():
        return None
    receipt = json.loads(path.read_text())
    if receipt["source_root"] != str(root):
        raise ValueError("compiler mirror belongs to another repository")
    for key, value in receipt["mappings"].items():
        if environment.get(key) != value["shared"]:
            raise ValueError("compiler mirror shared-cache configuration changed")
        local = Path(value["local"])
        if not local.is_dir():
            raise ValueError("compiler mirror missing")
        environment[key] = str(local)
        # Newly shared top-level keys fall back to the durable tree until refreshed.
        branches = (
            [""]
            if key == "TORCHINDUCTOR_CACHE_DIR"
            else ["torch_compile_cache", "torch_compile_cache/torch_aot_compile"]
        )
        for relative in branches:
            shared_branch = Path(value["shared"]) / relative
            local_branch = local / relative
            if not shared_branch.is_dir():
                continue
            local_branch.mkdir(parents=True, exist_ok=True)
            for child in shared_branch.iterdir():
                destination = local_branch / child.name
                if child.is_dir() and not destination.exists():
                    destination.symlink_to(child, target_is_directory=True)
    snapshot = target / "compiler_cache/snapshot.json"
    if not snapshot.exists():
        snapshot.write_text(
            json.dumps(
                {k: inventory(Path(v["local"])) for k, v in receipt["mappings"].items()}
            )
            + "\n"
        )
    receipt["snapshot"] = str(snapshot)
    return {"manifest_sha256": sha(path), **receipt}


def persist_compiler_mirror(receipt: dict | None) -> dict | None:
    """Copy new/updated immutable entries back; never delete durable cache keys."""
    if receipt is None:
        return None
    started = time.monotonic()
    before = json.loads(Path(receipt["snapshot"]).read_text())
    after, count = {}, 0
    for key, value in receipt["mappings"].items():
        after[key] = inventory(Path(value["local"]))
        changed = [
            p for p, stamp in after[key].items() if before.get(key, {}).get(p) != stamp
        ]
        count += len(changed)
        if changed:
            with tempfile.NamedTemporaryFile() as listing:
                listing.write(b"\0".join(p.encode() for p in changed) + b"\0")
                listing.flush()
                subprocess.run(
                    [
                        "rsync",
                        "-a",
                        "--update",
                        "--from0",
                        "--files-from=" + listing.name,
                        value["local"] + "/",
                        value["shared"] + "/",
                    ],
                    check=True,
                )
    Path(receipt["snapshot"]).write_text(json.dumps(after) + "\n")
    return {
        "completed": True,
        "changed_entries": count,
        "seconds": time.monotonic() - started,
    }
