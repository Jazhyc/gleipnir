#!/usr/bin/env python3
"""Rebuild the frozen ID and OOD monitoring evaluation sets from upstream.

The upstream datasets declare no license, so we share this builder instead of
the data. It downloads the pinned upstream files, runs the original
preparation steps and checks every output against its frozen SHA256.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.tool_trajectory_monitoring import (  # noqa: E402
    prepare_qwen_reasoning_id_benchmark as id_source,
)
from experiments.tool_trajectory_monitoring import (  # noqa: E402
    prepare_teacher_ood_cache as ood_source,
)
from scripts.prepare_hf_adapter_release import sha256_file  # noqa: E402

MODULE = "experiments.tool_trajectory_monitoring"
ID_SOURCE_DIRECTORY = Path("data/tool_trajectory_monitoring/source/id_evaluation")
ID_STEPS = (
    f"{MODULE}.prepare_qwen_reasoning_id_benchmark",
    f"{MODULE}.prepare_teacher_id_cache",
    f"{MODULE}.prepare_distillation_id",
)
OOD_STEPS = (
    f"{MODULE}.prepare_teacher_ood_cache",
    f"{MODULE}.prepare_distillation_ood",
)
DISTILLATION_ID = Path("data/tool_trajectory_monitoring/distillation_id/prompts.jsonl")
ID_OUTPUT = Path("data/id_cot_only_evaluation/prompts.jsonl")
ID_MANIFEST = Path("data/id_cot_only_evaluation/evaluation.manifest.json")
# Frozen outputs: docs/decisions/cot_removed_id_validation.md and
# docs/decisions/frozen_ood_comparability.md.
EXPECTED = {
    "id": {
        DISTILLATION_ID: (
            "ecc8b204be2505d2600c38d1747b87e2ef3c8a53ecd51908a79ad8e86e102389"
        ),
        ID_OUTPUT: "7a4ec604740b4b3b7bd70aafd4473b1b6bfcbf92706bcd23b90ed8df2c4bc1b4",
        ID_MANIFEST: (
            "abcc8267a77f6eff0a252ea846a6a59283e8a027cc860210e3984064eec611f1"
        ),
    },
    "ood": {
        ood_source.DEFAULT_OUTPUT: (
            "fa4b0d917e51eb91a52efa93cddf0eb154b88993269c97cfdb161c42680136a9"
        ),
        Path("data/tool_trajectory_monitoring/distillation_ood/prompts.jsonl"): (
            "ee2baa18a5d36cda8f9d66df642b6f3c106f92ae8ce2813acca36fa9f0e4acd2"
        ),
    },
}


def upstream_files(suite: str) -> list[tuple[str, str, str, str | None, Path]]:
    """(repo, revision, file, expected SHA256, local path) for each upstream file."""
    if suite == "id":
        return [
            (
                id_source.STRIDE_REPO,
                id_source.STRIDE_REVISION,
                id_source.STRIDE_FILE,
                id_source.STRIDE_SHA256,
                ID_SOURCE_DIRECTORY / "stride_test.parquet",
            ),
            (
                id_source.GLOOM_REPO,
                id_source.GLOOM_REVISION,
                id_source.GLOOM_FILE,
                id_source.GLOOM_SHA256,
                ID_SOURCE_DIRECTORY / "gloom_exfiltration.parquet",
            ),
        ]
    files = [
        (source.repo_file, source.sha256, source.filename)
        for source in ood_source.OOD_SOURCES
    ]
    files.append(
        (
            ood_source.FILTER_LOG_FILE,
            ood_source.FILTER_LOG_SHA256,
            Path(ood_source.FILTER_LOG_FILE).name,
        )
    )
    return [
        (
            ood_source.SOURCE_REPO,
            ood_source.SOURCE_REVISION,
            repo_file,
            expected,
            ood_source.SOURCE_DIRECTORY / name,
        )
        for repo_file, expected, name in files
    ]


def fetch(suite: str) -> None:
    for repo, revision, repo_file, expected, path in upstream_files(suite):
        if not path.exists():
            downloaded = hf_hub_download(
                repo, repo_file, repo_type="dataset", revision=revision
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(downloaded, path)
        if sha256_file(path) != expected:
            raise ValueError(f"upstream checksum drift: {path}")


def verify(suite: str) -> dict[str, str]:
    mismatched = {
        str(path): sha256_file(path) if path.exists() else "missing"
        for path, expected in EXPECTED[suite].items()
        if not path.exists() or sha256_file(path) != expected
    }
    if mismatched:
        raise ValueError(f"{suite} output differs from the frozen set: {mismatched}")
    return {str(path): expected for path, expected in EXPECTED[suite].items()}


def build(suite: str) -> dict[str, str]:
    fetch(suite)
    for module in ID_STEPS if suite == "id" else OOD_STEPS:
        subprocess.run([sys.executable, "-m", module], check=True, cwd=ROOT)
    if suite == "id":
        from experiments.id_cot_only_evaluation.prepare import write_evaluation

        write_evaluation(DISTILLATION_ID, ID_OUTPUT, ID_MANIFEST)
    return verify(suite)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("id", "ood", "all"))
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="check existing outputs without downloading or rebuilding",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    suites = ("id", "ood") if args.suite == "all" else (args.suite,)
    result = {
        suite: verify(suite) if args.verify_only else build(suite) for suite in suites
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
