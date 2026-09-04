#!/usr/bin/env python3
"""Materialize the selective-compilation candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.monitoring_gradient_checkpointing.prepare import selection_row
from experiments.monitoring_lr_sweep.prepare import (
    atomic_write_json,
    atomic_write_jsonl,
    read_jsonl,
    sha256_file,
)
from experiments.monitoring_selective_checkpointing.prepare import (
    make_job as make_checkpointing_job,
)
from experiments.monitoring_selective_compile.core import (
    COMPILE_BACKEND,
    COMPILE_MODE,
    COMPILE_POLICY,
    JOB_NAME,
    SOFT_TARGETS_SHA256,
    STUDENT_ROWS_SHA256,
    validate_jobs,
)
from experiments.monitoring_training_throughput.core import stable_stratified_selection

DEFAULT_DATA_DIR = Path("data/tool_trajectory_monitoring/distillation_scaling")
DEFAULT_RESULT_DIR = Path("results/monitoring_selective_compile")


def make_job(data_dir: Path, result_dir: Path, selection: Path) -> dict[str, Any]:
    """Adapt the selective-checkpointing job with only compilation enabled."""
    base = make_checkpointing_job(data_dir, result_dir, selection)
    output = result_dir / "runs" / JOB_NAME
    job = {
        **base,
        "job_name": JOB_NAME,
        "design_role": "selective_compilation_candidate",
        "selective_torch_compile_policy": COMPILE_POLICY,
        "selective_torch_compile_backend": COMPILE_BACKEND,
        "selective_torch_compile_mode": COMPILE_MODE,
        "selective_torch_compile_dynamic": True,
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
    selection = result_dir / "selections" / "matched-640.jsonl"
    selected = stable_stratified_selection(records)
    atomic_write_jsonl(selection, [selection_row(row) for row in selected])
    longest_path = result_dir / "selections" / "preflight-longest-32.jsonl"
    longest = sorted(
        records,
        key=lambda row: (int(row["student_direct_tokens"]), str(row["index"])),
        reverse=True,
    )[:32]
    atomic_write_jsonl(longest_path, [selection_row(row) for row in longest])
    job = make_job(data_dir, result_dir, selection)
    jobs_path = result_dir / "jobs.jsonl"
    atomic_write_jsonl(jobs_path, [job])
    manifest = {
        "campaign_id": "gleipnir4b-monitoring-selective-compile-v1",
        "jobs_sha256": sha256_file(jobs_path),
        "selection_sha256": sha256_file(selection),
        "preflight_selection_sha256": sha256_file(longest_path),
    }
    atomic_write_json(result_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
