#!/usr/bin/env python3
"""Materialize the exact matched-example throughput jobs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.monitoring_lr_sweep.prepare import (
    atomic_write_json,
    atomic_write_jsonl,
    read_jsonl,
    sha256_file,
)
from experiments.monitoring_training_throughput.core import (
    CONDITIONS,
    EFFECTIVE_BATCH_SIZE,
    LEARNING_RATE,
    LORA_ALPHA,
    MAX_LENGTH,
    MODEL_ID,
    MODEL_REVISION,
    OPTIMIZER_STEPS,
    RANK,
    SEED,
    SELECTION_ROWS,
    SOFT_TARGETS_SHA256,
    STUDENT_ROWS_SHA256,
    stable_stratified_selection,
    validate_jobs,
)

DEFAULT_DATA_DIR = Path("data/tool_trajectory_monitoring/distillation_scaling")
DEFAULT_RESULT_DIR = Path("results/monitoring_training_throughput")


def make_jobs(
    data_dir: Path, result_dir: Path, selection_path: Path
) -> list[dict[str, Any]]:
    """Build the three frozen throughput jobs."""
    common = {
        "capacity_kind": "lora",
        "data_scope": "monitoring_only",
        "deception_rows": 0,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "seed": SEED,
        "target": "kimi_soft",
        "rank": RANK,
        "lora_alpha": LORA_ALPHA,
        "max_length": MAX_LENGTH,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "optimizer": "adamw",
        "learning_rate": LEARNING_RATE,
        "lr_scheduler_type": "linear",
        "warmup_ratio": 0.03,
        "weight_decay": 0.0,
        "num_train_epochs": -1,
        "max_steps": OPTIMIZER_STEPS,
        "soft_loss_weight": 1.0,
        "direct_loss_weight": 0.0,
        "require_causal_conv1d": True,
        "fla_disable_backend_dispatch": True,
        "triton_version": "3.7.1",
        "train_rows": SELECTION_ROWS,
        "selection_manifest": selection_path.as_posix(),
        "selection_sha256": sha256_file(selection_path),
        "student_rows": (data_dir / "student_rows.jsonl").as_posix(),
        "soft_targets": (data_dir / "soft_targets.jsonl").as_posix(),
        "student_rows_sha256": STUDENT_ROWS_SHA256,
        "soft_targets_sha256": SOFT_TARGETS_SHA256,
        "save_steps": OPTIMIZER_STEPS,
    }
    jobs = []
    for condition in CONDITIONS:
        output = result_dir / "runs" / str(condition["job_name"])
        jobs.append(
            {
                **common,
                **condition,
                "design_role": (
                    "stopped_sweep_baseline"
                    if condition["job_name"] == "mb2-random"
                    else "throughput_candidate"
                ),
                "output_dir": output.as_posix(),
                "causal_adapter_dir": (output / "causal_adapter").as_posix(),
                "model_dir": (output / "model").as_posix(),
            }
        )
    validate_jobs(jobs)
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    result_dir = args.result_dir.resolve()
    student_path = data_dir / "student_rows.jsonl"
    soft_path = data_dir / "soft_targets.jsonl"
    if sha256_file(student_path) != STUDENT_ROWS_SHA256:
        raise ValueError("student-row checksum drifted")
    if sha256_file(soft_path) != SOFT_TARGETS_SHA256:
        raise ValueError("soft-target checksum drifted")
    records = read_jsonl(student_path)
    if len(records) != 8_688:
        raise ValueError("monitoring row count drifted")
    selected = stable_stratified_selection(records)
    selection_rows = [
        {
            "dataset": row["dataset"],
            "index": row["index"],
            "label": row["label"],
            "source_dataset": row["source_dataset"],
            "student_direct_tokens": row["student_direct_tokens"],
            "trajectory_sha256": row["trajectory_sha256"],
        }
        for row in selected
    ]
    selection_path = result_dir / "selections" / "matched-640.jsonl"
    atomic_write_jsonl(selection_path, selection_rows)
    jobs = make_jobs(data_dir, result_dir, selection_path)
    jobs_path = result_dir / "jobs.jsonl"
    atomic_write_jsonl(jobs_path, jobs)
    lengths = sorted(int(row["student_direct_tokens"]) for row in selected)
    manifest = {
        "campaign_id": "gleipnir4b-monitoring-training-throughput-v1",
        "selection_rows": len(selected),
        "selection_sha256": sha256_file(selection_path),
        "jobs_sha256": sha256_file(jobs_path),
        "source_label_counts": dict(
            sorted(
                Counter(
                    f"{row['source_dataset']}:{int(row['label'])}" for row in selected
                ).items()
            )
        ),
        "direct_token_lengths": {
            "minimum": lengths[0],
            "median": lengths[len(lengths) // 2],
            "p90": lengths[int(0.9 * (len(lengths) - 1))],
            "maximum": lengths[-1],
            "total": sum(lengths),
        },
        "jobs": [job["job_name"] for job in jobs],
    }
    atomic_write_json(result_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
