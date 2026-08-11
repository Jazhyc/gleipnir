#!/usr/bin/env python3
"""Rebase text-only Qwen3.5 LoRAs onto its multimodal language-model path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.data_scaling.prepare import read_jsonl  # noqa: E402, I001
from gleipnir.qwen35_adapter_rebase import (  # noqa: E402
    DESTINATION_PREFIX,
    MANIFEST_NAME,
    SOURCE_PREFIX,
    rebase_adapter,
    rebase_key,
)

__all__ = [
    "DESTINATION_PREFIX",
    "MANIFEST_NAME",
    "SOURCE_PREFIX",
    "rebase_adapter",
    "rebase_key",
]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jobs", type=Path, default=Path("results/data_scaling/lambda_jobs.jsonl")
    )
    parser.add_argument(
        "--output-jobs",
        type=Path,
        default=Path("results/data_scaling/lambda_jobs_rebased.jsonl"),
    )
    parser.add_argument("--directory-name", default="adapter_image_text_to_text")
    args = parser.parse_args()

    jobs = read_jsonl(args.jobs.resolve())
    rebased_jobs = []
    for job in jobs:
        source_dir = Path(job["adapter_dir"])
        destination_dir = source_dir.parent / args.directory_name
        manifest = rebase_adapter(source_dir, destination_dir)
        rebased = dict(job)
        rebased.update(
            {
                "source_adapter_dir": source_dir.resolve().as_posix(),
                "adapter_dir": destination_dir.resolve().as_posix(),
                "adapter_rebase_manifest": (
                    destination_dir / MANIFEST_NAME
                ).resolve().as_posix(),
                "adapter_rebase_source_sha256": manifest["source_sha256"],
                "adapter_rebase_destination_sha256": manifest[
                    "destination_sha256"
                ],
            }
        )
        rebased_jobs.append(rebased)
        print(f"rebased {job['job_name']} -> {destination_dir}", flush=True)
    write_jsonl(args.output_jobs.resolve(), rebased_jobs)


if __name__ == "__main__":
    main()
