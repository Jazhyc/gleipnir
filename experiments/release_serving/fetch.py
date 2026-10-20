"""Download a release target's merged BF16 checkpoint and verify every file.

The checkpoint goes to the recorded merged path (part of the compiler-cache
key). Set ``GLEIPNIR_MODEL_CACHE`` to a persistent directory to keep downloads
across container restarts; the recorded path then links to it.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path

from experiments.release_serving.targets import ROOT, TARGETS
from gleipnir.data.monitoring import file_hash


def expected_files(merged_artifact: str) -> dict[str, str]:
    value = json.loads((ROOT / merged_artifact).read_text())["files_sha256"]
    return ast.literal_eval(value) if isinstance(value, str) else value


def verified(directory: Path, expected: dict[str, str]) -> bool:
    return all(
        (directory / name).is_file() and file_hash(directory / name) == digest
        for name, digest in expected.items()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=sorted(TARGETS))
    args = parser.parse_args()
    target = TARGETS[args.model]
    destination = target.merged_path
    expected = expected_files(target.merged_artifact)
    cache = os.environ.get("GLEIPNIR_MODEL_CACHE")
    local = Path(cache) / destination.name if cache else destination
    if cache and not destination.exists():
        local.mkdir(parents=True, exist_ok=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(local, target_is_directory=True)
    if verified(local, expected):
        print(
            f"gleipnir_fetch_cached {target.merged_repo}@{target.version}", flush=True
        )
        return
    from huggingface_hub import snapshot_download

    snapshot_download(
        target.merged_repo,
        revision=target.version,
        local_dir=local,
        allow_patterns=sorted(expected),
    )
    if not verified(local, expected):
        raise ValueError("downloaded checkpoint does not match the release hashes")
    print(f"gleipnir_fetch_done {target.merged_repo}@{target.version}", flush=True)


if __name__ == "__main__":
    main()
