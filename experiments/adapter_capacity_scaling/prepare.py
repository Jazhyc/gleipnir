#!/usr/bin/env python3
"""Validate full-data inputs and prepare the rank/full-finetuning jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.adapter_capacity_scaling.core import (  # noqa: E402
    DEFAULT_RANKS,
    DEFAULT_SEEDS,
    validate_ranks,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = "".join(
        json.dumps(record, sort_keys=True) + "\n" for record in records
    )
    path.write_text(contents)


def identity(record: dict[str, Any]) -> tuple[str, Any]:
    return str(record["dataset"]), record["index"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--student-rows",
        type=Path,
        default=Path("data/data_scaling/student_rows.jsonl"),
    )
    parser.add_argument(
        "--soft-targets",
        type=Path,
        default=Path("data/data_scaling/soft_targets.jsonl"),
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("data/data_scaling/validation.jsonl"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/adapter_capacity_scaling")
    )
    parser.add_argument("--ranks", nargs="+", type=int, default=DEFAULT_RANKS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    args = parser.parse_args()

    student_path = args.student_rows.resolve()
    soft_path = args.soft_targets.resolve()
    validation_path = args.validation.resolve()
    output_dir = args.output_dir.resolve()
    ranks = validate_ranks(args.ranks)
    seeds = sorted(set(args.seeds))
    if not seeds:
        raise ValueError("at least one seed is required")

    records = read_jsonl(student_path)
    targets = read_jsonl(soft_path)
    validation = read_jsonl(validation_path)
    if len(records) != 13149 or len(targets) != len(records):
        raise ValueError("expected matching 13,149-row student and target artifacts")
    record_ids = [identity(record) for record in records]
    target_by_id = {identity(record): record for record in targets}
    if len(set(record_ids)) != len(records) or len(target_by_id) != len(targets):
        raise ValueError("training or target artifact contains duplicate identities")
    for record in records:
        target = target_by_id.get(identity(record))
        if target is None or int(target["label"]) != int(record["label"]):
            raise ValueError(
                f"missing or label-inconsistent target for {identity(record)}"
            )
    overlap = set(record_ids) & {identity(record) for record in validation}
    if overlap:
        raise ValueError(f"training/validation identity leakage: {sorted(overlap)[:3]}")

    jobs = []
    for seed in seeds:
        for rank in ranks:
            job_name = f"seed{seed}-r{rank:03d}"
            job_dir = output_dir / "runs" / job_name
            jobs.append(
                {
                    "job_name": job_name,
                    "capacity_kind": "lora",
                    "seed": seed,
                    "rank": rank,
                    "lora_alpha": 2 * rank,
                    "train_rows": len(records),
                    "student_rows": student_path.as_posix(),
                    "soft_targets": soft_path.as_posix(),
                    "validation": validation_path.as_posix(),
                    "output_dir": job_dir.as_posix(),
                    "model_dir": (job_dir / "model").as_posix(),
                }
            )
        job_name = f"seed{seed}-full"
        job_dir = output_dir / "runs" / job_name
        jobs.append(
            {
                "job_name": job_name,
                "capacity_kind": "full",
                "seed": seed,
                "rank": None,
                "lora_alpha": None,
                "train_rows": len(records),
                "student_rows": student_path.as_posix(),
                "soft_targets": soft_path.as_posix(),
                "validation": validation_path.as_posix(),
                "output_dir": job_dir.as_posix(),
                "model_dir": (job_dir / "model").as_posix(),
            }
        )
    jobs_path = output_dir / "jobs.jsonl"
    write_jsonl(jobs_path, jobs)

    upstream = Path("results/data_scaling/manifest.json").resolve()
    manifest = {
        "hypothesis": (
            "full-data monitor quality follows a bounded power law in LoRA rank, "
            "with full fine-tuning as an external capacity endpoint"
        ),
        "student_rows": student_path.as_posix(),
        "student_rows_sha256": sha256_file(student_path),
        "soft_targets": soft_path.as_posix(),
        "soft_targets_sha256": sha256_file(soft_path),
        "validation": validation_path.as_posix(),
        "validation_sha256": sha256_file(validation_path),
        "training_rows": len(records),
        "validation_rows": len(validation),
        "validation_datasets": len({record["dataset"] for record in validation}),
        "train_validation_overlap": 0,
        "ranks": ranks,
        "lora_alpha_over_rank": 2.0,
        "seeds": seeds,
        "jobs": len(jobs),
        "lora_jobs": len(ranks) * len(seeds),
        "full_finetuning_jobs": len(seeds),
        "jobs_path": jobs_path.as_posix(),
        "upstream_data_scaling_manifest": (
            upstream.as_posix() if upstream.is_file() else None
        ),
        "upstream_data_scaling_manifest_sha256": (
            sha256_file(upstream) if upstream.is_file() else None
        ),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"prepared {len(jobs)} jobs: {len(ranks) * len(seeds)} LoRA and "
        f"{len(seeds)} full fine-tunes; manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()
