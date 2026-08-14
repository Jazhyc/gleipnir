#!/usr/bin/env python3
"""Prepare the frozen structural training-procedure screen."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.adapter_capacity_scaling.prepare import (  # noqa: E402
    identity,
    read_jsonl,
    sha256_file,
    write_jsonl,
)
from experiments.training_procedure_screen.core import (  # noqa: E402
    completed_variant,
    screen_variants,
    validate_screen_jobs,
)


def selection_row(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": record["dataset"],
        "index": record["index"],
        "label": int(record["label"]),
    }


def base_job(
    variant: dict[str, Any],
    *,
    train_rows: int,
    student_path: Path,
    soft_path: Path,
    selection_path: Path,
    development_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    completed = completed_variant(variant)
    job_name = str(completed["job_name"])
    job_dir = output_dir / "runs" / job_name
    effective_batch = int(completed["micro_batch_size"]) * int(
        completed["gradient_accumulation_steps"]
    )
    micro_batches_per_epoch = math.ceil(
        train_rows / int(completed["micro_batch_size"])
    )
    optimizer_steps_per_epoch = math.ceil(
        micro_batches_per_epoch / int(completed["gradient_accumulation_steps"])
    )
    checkpoint_steps = math.ceil(optimizer_steps_per_epoch / 2)
    return {
        **completed,
        "capacity_kind": "lora",
        "config_path": "../training_procedure_screen",
        "config_name": "config",
        "rank": 128,
        "lora_alpha": 256,
        "lora_dropout": 0.0,
        "optimizer": "adamw",
        "learning_rate": 5e-5,
        "lr_scheduler_type": "linear",
        "warmup_ratio": 0.03,
        "num_train_epochs": 2.0,
        "weight_decay": 0.0,
        "dataset_sampling": "proportional",
        "soft_target_logit_scale": 1.0,
        "save_strategy": "steps",
        "save_steps": checkpoint_steps,
        "save_total_limit": 6,
        "save_only_model": True,
        "train_rows": train_rows,
        "selection_manifest": selection_path.as_posix(),
        "selection_sha256": sha256_file(selection_path),
        "student_rows": student_path.as_posix(),
        "soft_targets": soft_path.as_posix(),
        "validation": development_path.as_posix(),
        "output_dir": job_dir.as_posix(),
        "causal_adapter_dir": (job_dir / "causal_adapter").as_posix(),
        "model_dir": (job_dir / "model").as_posix(),
        "effective_batch_size": effective_batch,
        "evaluation_base": completed.get("evaluation_base", "bf16"),
        "serving_backend": completed.get("serving_backend", "vllm_candidate"),
    }


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
        "--development",
        type=Path,
        default=Path("data/optimization_regularization_screen/development.jsonl"),
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/training_procedure_screen")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/training_procedure_screen")
    )
    args = parser.parse_args()

    student_path = args.student_rows.resolve()
    soft_path = args.soft_targets.resolve()
    development_path = args.development.resolve()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    records = read_jsonl(student_path)
    targets = read_jsonl(soft_path)
    development = read_jsonl(development_path)
    if len(records) != 13149 or len(targets) != len(records):
        raise ValueError("expected matching 13,149-row student and target artifacts")
    target_by_id = {identity(record): record for record in targets}
    if len(target_by_id) != len(targets):
        raise ValueError("soft-target artifact contains duplicate identities")
    for record in records:
        target = target_by_id.get(identity(record))
        if target is None or int(target["label"]) != int(record["label"]):
            raise ValueError(f"missing or inconsistent target for {identity(record)!r}")
    development_ids = {identity(record) for record in development}
    if len(development) != 1972 or len(development_ids) != len(development):
        raise ValueError("expected the fixed unique 1,972-row development artifact")
    training = [record for record in records if identity(record) not in development_ids]
    if len(training) != 11177:
        raise ValueError(f"expected 11,177 training rows, got {len(training)}")

    data_dir.mkdir(parents=True, exist_ok=True)
    selection_path = data_dir / "train_full_tuning_pool.jsonl"
    write_jsonl(selection_path, [selection_row(record) for record in training])
    jobs = [
        base_job(
            variant,
            train_rows=len(training),
            student_path=student_path,
            soft_path=soft_path,
            selection_path=selection_path,
            development_path=development_path,
            output_dir=output_dir,
        )
        for variant in screen_variants()
    ]
    validate_screen_jobs(jobs)
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs_path = output_dir / "jobs.jsonl"
    write_jsonl(jobs_path, jobs)
    manifest = {
        "hypothesis": (
            "base precision, adapter initialization/parameterization/targets, "
            "loss geometry, group weighting, batch noise, checkpoint averaging, "
            "or seed ensembling improves direct binary monitor generalization"
        ),
        "intervention": "seventeen single-factor cells plus three baseline seeds",
        "jobs": len(jobs),
        "job_names": [job["job_name"] for job in jobs],
        "seeds": [0, 1, 2],
        "training_rows": len(training),
        "development_rows": len(development),
        "training_development_overlap": 0,
        "development_lineage_grouping": (
            "unavailable: source rows expose dataset and row identity but no "
            "conversation or generator lineage field"
        ),
        "rank": 128,
        "lora_alpha": 256,
        "optimizer": "adamw",
        "student_rows_sha256": sha256_file(student_path),
        "soft_targets_sha256": sha256_file(soft_path),
        "selection_sha256": sha256_file(selection_path),
        "development_sha256": sha256_file(development_path),
        "competition_validation_used": False,
        "final_test_used": False,
        "primary_metric": "macro_auroc",
        "metric_constraints": {
            "macro_balanced_accuracy_max_regression": 0.002,
            "macro_brier_max_regression": 0.002,
        },
        "selection_rule": (
            "rank seed-0 single-factor cells against baseline seed 0 after metric "
            "constraints; require a positive AUROC delta and then seeds 1/2 before "
            "promotion"
        ),
        "diagnostics": [
            "all retained half-epoch checkpoints",
            "baseline seed logit ensemble",
            "baseline checkpoint logit averages",
            "cross-fitted global threshold and Platt calibration",
        ],
        "stop_condition": (
            "stop on split overlap, design drift, missing FLA, failed risk-cell "
            "preflight, absent metadata, or incomplete causal evaluation"
        ),
        "jobs_path": jobs_path.as_posix(),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"prepared {len(jobs)} jobs with {len(training)} training and "
        f"{len(development)} development rows; manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()
