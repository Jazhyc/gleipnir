#!/usr/bin/env python3
"""Materialize matched jobs for the gradient-checkpointing screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.monitoring_gradient_checkpointing.core import (
    EFFECTIVE_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    JOB_NAME,
    LEARNING_RATE,
    LORA_ALPHA,
    MAX_LENGTH,
    MICRO_BATCH_SIZE,
    MODEL_ID,
    MODEL_REVISION,
    OPTIMIZER_STEPS,
    RANK,
    SEED,
    SELECTION_ROWS,
    SOFT_TARGETS_SHA256,
    STUDENT_ROWS_SHA256,
    validate_jobs,
)
from experiments.monitoring_lr_sweep.prepare import (
    atomic_write_json,
    atomic_write_jsonl,
    read_jsonl,
    sha256_file,
)
from experiments.monitoring_training_throughput.core import stable_stratified_selection
from gleipnir.monitoring_systems_screen import (
    selection_manifest_row as selection_row,
)

DEFAULT_DATA_DIR = Path("data/tool_trajectory_monitoring/distillation_scaling")
DEFAULT_RESULT_DIR = Path("results/monitoring_gradient_checkpointing")


def make_job(data_dir: Path, result_dir: Path, selection_path: Path) -> dict[str, Any]:
    """Build the one frozen checkpointing-off job."""
    output = result_dir / "runs" / JOB_NAME
    job = {
        "job_name": JOB_NAME,
        "design_role": "throughput_candidate",
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
        "micro_batch_size": MICRO_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "gradient_checkpointing": False,
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
        "train_sampling_strategy": "random",
        "train_rows": SELECTION_ROWS,
        "selection_manifest": selection_path.as_posix(),
        "selection_sha256": sha256_file(selection_path),
        "student_rows": (data_dir / "student_rows.jsonl").as_posix(),
        "soft_targets": (data_dir / "soft_targets.jsonl").as_posix(),
        "student_rows_sha256": STUDENT_ROWS_SHA256,
        "soft_targets_sha256": SOFT_TARGETS_SHA256,
        "save_steps": OPTIMIZER_STEPS,
        "output_dir": output.as_posix(),
        "causal_adapter_dir": (output / "causal_adapter").as_posix(),
        "model_dir": (output / "model").as_posix(),
    }
    validate_jobs([job])
    return job


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
    selected = stable_stratified_selection(records)
    selection_path = result_dir / "selections" / "matched-640.jsonl"
    longest_path = result_dir / "selections" / "preflight-longest-32.jsonl"
    atomic_write_jsonl(selection_path, [selection_row(row) for row in selected])
    longest = sorted(
        records,
        key=lambda row: (int(row["student_direct_tokens"]), str(row["index"])),
        reverse=True,
    )[:32]
    atomic_write_jsonl(longest_path, [selection_row(row) for row in longest])
    job = make_job(data_dir, result_dir, selection_path)
    jobs_path = result_dir / "jobs.jsonl"
    atomic_write_jsonl(jobs_path, [job])
    manifest = {
        "campaign_id": "gleipnir4b-monitoring-gradient-checkpointing-v1",
        "jobs_sha256": sha256_file(jobs_path),
        "selection_sha256": sha256_file(selection_path),
        "preflight_selection_sha256": sha256_file(longest_path),
        "selection_rows": len(selected),
        "preflight_rows": len(longest),
        "maximum_direct_tokens": int(longest[0]["student_direct_tokens"]),
    }
    atomic_write_json(result_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
