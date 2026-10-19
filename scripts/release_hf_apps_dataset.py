#!/usr/bin/env python3
"""Stage and upload the APPS injection benchmark with our honest controls."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_hf_adapter_release import (  # noqa: E402
    sha256_file,
    staged_files,
    write_json,
)
from scripts.upload_hf_adapter_release import load_manifest  # noqa: E402

REPO_NAME = "gleipnir-apps-injection-evals"
CARD = ROOT / "data_releases/hf" / REPO_NAME / "README.md"
DEFAULT_OUTPUT = ROOT / "results/huggingface_release"
DEFAULT_CONTROLS_ROOT = ROOT.parent / "monitor-injection-evals"
UPSTREAM_REPO = "mjkenney/monitor-injection-evals"
UPSTREAM_REVISION = "58e722bf9912903d56a334f873e069aa17ccb7ad"
# Honest-control builder: scripts/build_honest_controls.py at this commit of
# github.com/mkenney2/monitor-injection-evals.
CONTROLS_COMMIT = "b55c2014d4c22ceea810b2639e6c0f672d83eccb"

# Upstream files, copied byte-for-byte; None means small metadata without a frozen hash.
UPSTREAM_FILES = {
    "data/apps/apps_eval.jsonl": (
        "d54a8677329278d641dbee91b1740a5fda11b9c2177a8c4fad8fe3f58ce276c5"
    ),
    "data/apps/manifest.json": None,
    "data/apps/prompts/eval_neutral.jsonl": (
        "339327a476f6834d8998500e2726263dfe3ce55bcc0a12648928abe9e99abfc0"
    ),
    "data/apps/prompts/eval_neutral.manifest.json": None,
    "data/apps/prompts/eval_injection_aware.jsonl": (
        "b376a701853d4189a300aaf3d310e949256f38b2619b28017d362c446c284eec"
    ),
    "data/apps/prompts/eval_injection_aware.manifest.json": None,
    "data/templates/transfer_gpt4.json": None,
    "data/templates/transfer_gpt41.json": None,
    "data/templates/transfer_gpt4o.json": None,
}
CONTROL_FILES = {
    "data/apps/honest_controls/eval_neutral.jsonl": (
        "71b337c070fc60faba0d04c54c5f0813498c45d60ab6b9c77b9b67639c6a70e1"
    ),
    "data/apps/honest_controls/eval_neutral.manifest.json": None,
    "data/apps/honest_controls/eval_injection_aware.jsonl": (
        "6c90d47d6c0fac4cdc2b97803e004a42d49cd7488c4c8f71bbb5f894ac802369"
    ),
    "data/apps/honest_controls/eval_injection_aware.manifest.json": None,
}
EXPECTED_ROWS = {
    "data/apps/prompts/eval_neutral.jsonl": 5184,
    "data/apps/prompts/eval_injection_aware.jsonl": 5184,
    "data/apps/honest_controls/eval_neutral.jsonl": 3930,
    "data/apps/honest_controls/eval_injection_aware.jsonl": 3930,
}


def copy_checked(source: Path, destination: Path, expected: str | None) -> None:
    if expected is not None and sha256_file(source) != expected:
        raise ValueError(f"frozen checksum drift: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def check_controls(staged: Path) -> None:
    """Each honest-control manifest must name its staged rows and parent benchmark."""
    for variant in ("neutral", "injection_aware"):
        base = staged / "data/apps/honest_controls"
        manifest = json.loads((base / f"eval_{variant}.manifest.json").read_text())
        rows = base / f"eval_{variant}.jsonl"
        if manifest["output"]["sha256"] != sha256_file(rows):
            raise ValueError(f"honest-control manifest mismatch: {variant}")
        source = manifest["source"]
        parent = f"data/apps/prompts/eval_{variant}.jsonl"
        if (
            source["hf_revision"] != UPSTREAM_REVISION
            or source["sha256"] != UPSTREAM_FILES[parent]
        ):
            raise ValueError(f"honest-control parent drift: {variant}")
    for relative, expected in EXPECTED_ROWS.items():
        with (staged / relative).open(encoding="utf-8") as handle:
            if sum(1 for _ in handle) != expected:
                raise ValueError(f"row count drift: {relative}")


def stage(output_root: Path, controls_root: Path) -> Path:
    destination = output_root / REPO_NAME
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite staged release: {destination}")
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{REPO_NAME}-", dir=output_root))
    try:
        shutil.copyfile(CARD, temporary / "README.md")
        for relative, expected in UPSTREAM_FILES.items():
            source = hf_hub_download(
                UPSTREAM_REPO,
                relative,
                repo_type="dataset",
                revision=UPSTREAM_REVISION,
            )
            copy_checked(Path(source), temporary / relative, expected)
        for relative, expected in CONTROL_FILES.items():
            copy_checked(controls_root / relative, temporary / relative, expected)
        check_controls(temporary)
        write_json(
            temporary / "release_manifest.json",
            {
                "controls_commit": CONTROLS_COMMIT,
                "display_name": "Gleipnir APPS injection evals",
                "files": staged_files(temporary),
                "format_version": 1,
                "repo_name": REPO_NAME,
                "upstream_repo": UPSTREAM_REPO,
                "upstream_revision": UPSTREAM_REVISION,
            },
        )
        temporary.replace(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def upload(release: Path, namespace: str, public: bool, allow_existing: bool) -> str:
    manifest = load_manifest(release)
    repo_id = f"{namespace}/{manifest['repo_name']}"
    api = HfApi()
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=not public,
        exist_ok=allow_existing,
    )
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=release,
        commit_message="Release APPS injection benchmark with honest controls",
    )
    return repo_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    prepare.add_argument("--controls-root", type=Path, default=DEFAULT_CONTROLS_ROOT)
    push = commands.add_parser("upload")
    push.add_argument("--release-dir", type=Path, default=DEFAULT_OUTPUT / REPO_NAME)
    push.add_argument("--namespace", required=True)
    push.add_argument("--public", action="store_true")
    push.add_argument("--allow-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        result = {"staged": str(stage(args.output_dir, args.controls_root))}
    else:
        result = {
            "uploaded": upload(
                args.release_dir.resolve(),
                args.namespace,
                args.public,
                args.allow_existing,
            )
        }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
