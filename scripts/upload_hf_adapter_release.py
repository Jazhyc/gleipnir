#!/usr/bin/env python3
"""Upload already-staged Gleipnir adapter releases to Hugging Face."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_hf_adapter_release import sha256_file  # noqa: E402


def load_manifest(release: Path) -> dict[str, Any]:
    path = release / "release_manifest.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("format_version") != 1:
        raise ValueError(f"invalid release manifest: {path}")
    expected = set(value.get("files", {})) | {"release_manifest.json"}
    actual = {
        item.relative_to(release).as_posix()
        for item in release.rglob("*")
        if item.is_file()
    }
    if actual != expected:
        raise ValueError(
            f"staged file set drifted for {release}: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    for relative, details in value["files"].items():
        path = release / relative
        if path.stat().st_size != int(details["bytes"]):
            raise ValueError(f"staged file size drifted: {path}")
        if sha256_file(path) != details["sha256"]:
            raise ValueError(f"staged file checksum drifted: {path}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--model", choices=("all", "4b", "9b"), default="all")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--allow-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suffixes = ("4B", "9B") if args.model == "all" else (args.model.upper(),)
    releases = [
        args.release_dir.resolve() / f"Gleipnir-{suffix}" for suffix in suffixes
    ]
    manifests = [load_manifest(release) for release in releases]
    api = HfApi()
    uploaded = []
    for release, manifest in zip(releases, manifests, strict=True):
        repo_id = f"{args.namespace}/{manifest['repo_name']}"
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=args.private,
            exist_ok=args.allow_existing,
        )
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=release,
            commit_message=f"Release {manifest['display_name']} research artifact",
        )
        uploaded.append(repo_id)
    print(json.dumps({"uploaded": uploaded}, sort_keys=True))


if __name__ == "__main__":
    main()
