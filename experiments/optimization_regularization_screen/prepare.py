#!/usr/bin/env python3
"""Prepare the frozen 16-cell optimization and regularization screen."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.adapter_capacity_scaling.core import (  # noqa: E402
    QLORA_EFFECTIVE_BATCH_SIZE,
    QLORA_GRADIENT_ACCUMULATION_STEPS,
    QLORA_MICRO_BATCH_SIZE,
)
from experiments.adapter_capacity_scaling.prepare import (  # noqa: E402
    identity,
    read_jsonl,
    sha256_file,
    write_jsonl,
)
from experiments.data_scaling.core import nested_stratified_selections  # noqa: E402
from experiments.optimization_regularization_screen.core import (  # noqa: E402
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
    checkpoint_steps: int,
    student_path: Path,
    soft_path: Path,
    selection_path: Path,
    development_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    job_name = str(variant["job_name"])
    job_dir = output_dir / "runs" / job_name
    job: dict[str, Any] = {
        "job_name": job_name,
        "intervention_family": variant["intervention_family"],
        "capacity_kind": "lora",
        "seed": 0,
        "rank": 128,
        "lora_alpha": 256,
        "optimizer": "adamw",
        "learning_rate": 5e-5,
        "lr_scheduler_type": "linear",
        "warmup_ratio": 0.03,
        "num_train_epochs": 2.0,
        "lora_dropout": 0.0,
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
        "training_recipe": "qlora-nf4-double-quant-bf16-fla-0.5.2",
        "micro_batch_size": QLORA_MICRO_BATCH_SIZE,
        "gradient_accumulation_steps": QLORA_GRADIENT_ACCUMULATION_STEPS,
        "effective_batch_size": QLORA_EFFECTIVE_BATCH_SIZE,
    }
    job.update(
        {key: value for key, value in variant.items() if key != "job_name"}
    )
    return job


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
        "--data-dir",
        type=Path,
        default=Path("data/optimization_regularization_screen"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/optimization_regularization_screen"),
    )
    parser.add_argument("--development-fraction", type=float, default=0.15)
    parser.add_argument("--training-fraction-of-pool", type=float, default=0.50)
    parser.add_argument("--development-seed", type=int, default=20260814)
    parser.add_argument("--training-seed", type=int, default=0)
    args = parser.parse_args()
    if not 0 < args.development_fraction < 0.5:
        raise ValueError("development fraction must be in (0, 0.5)")
    if not 0 < args.training_fraction_of_pool < 1:
        raise ValueError("training fraction of tuning pool must be in (0, 1)")

    student_path = args.student_rows.resolve()
    soft_path = args.soft_targets.resolve()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    records = read_jsonl(student_path)
    targets = read_jsonl(soft_path)
    if len(records) != 13149 or len(targets) != len(records):
        raise ValueError("expected matching 13,149-row student and target artifacts")
    target_by_id = {identity(record): record for record in targets}
    if len(target_by_id) != len(targets):
        raise ValueError("soft-target artifact contains duplicate identities")
    for record in records:
        target = target_by_id.get(identity(record))
        if target is None or int(target["label"]) != int(record["label"]):
            raise ValueError(f"missing or inconsistent target for {identity(record)!r}")

    development = nested_stratified_selections(
        records, [args.development_fraction], args.development_seed
    )[args.development_fraction]
    development_ids = {identity(record) for record in development}
    tuning_pool = [
        record for record in records if identity(record) not in development_ids
    ]
    training = nested_stratified_selections(
        tuning_pool,
        [args.training_fraction_of_pool],
        args.training_seed,
    )[args.training_fraction_of_pool]
    training_ids = {identity(record) for record in training}
    if training_ids & development_ids:
        raise AssertionError("optimization training and development splits overlap")

    data_dir.mkdir(parents=True, exist_ok=True)
    selection_path = data_dir / "train_seed0_pool_f050.jsonl"
    development_path = data_dir / "development.jsonl"
    write_jsonl(selection_path, [selection_row(record) for record in training])
    write_jsonl(development_path, development)
    micro_batches_per_epoch = math.ceil(len(training) / QLORA_MICRO_BATCH_SIZE)
    optimizer_steps_per_epoch = math.ceil(
        micro_batches_per_epoch / QLORA_GRADIENT_ACCUMULATION_STEPS
    )
    checkpoint_steps = math.ceil(optimizer_steps_per_epoch / 2)
    jobs = [
        base_job(
            variant,
            train_rows=len(training),
            checkpoint_steps=checkpoint_steps,
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

    dataset_counts = Counter(str(record["dataset"]) for record in training)
    manifest = {
        "hypothesis": (
            "trajectory, regularization, target smoothing, or macro-aligned "
            "dataset sampling improves rank-128 AdamW soft distillation"
        ),
        "intervention": "fifteen single-factor cells around one shared baseline",
        "jobs": len(jobs),
        "job_names": [job["job_name"] for job in jobs],
        "training_seed": args.training_seed,
        "development_seed": args.development_seed,
        "development_fraction": args.development_fraction,
        "development_rows": len(development),
        "tuning_pool_rows": len(tuning_pool),
        "training_fraction_of_tuning_pool": args.training_fraction_of_pool,
        "training_rows": len(training),
        "training_development_overlap": 0,
        "development_lineage_grouping": (
            "unavailable: source rows expose dataset and row identity but no "
            "conversation or generator lineage field"
        ),
        "training_dataset_counts": dict(sorted(dataset_counts.items())),
        "training_dataset_min_rows": min(dataset_counts.values()),
        "training_dataset_max_rows": max(dataset_counts.values()),
        "rank": 128,
        "lora_alpha": 256,
        "optimizer": "adamw",
        "checkpoint_interval": "approximately 0.5 epoch",
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "checkpoint_steps": checkpoint_steps,
        "student_rows": student_path.as_posix(),
        "student_rows_sha256": sha256_file(student_path),
        "soft_targets": soft_path.as_posix(),
        "soft_targets_sha256": sha256_file(soft_path),
        "selection_manifest": selection_path.as_posix(),
        "selection_sha256": sha256_file(selection_path),
        "development": development_path.as_posix(),
        "development_sha256": sha256_file(development_path),
        "jobs_path": jobs_path.as_posix(),
        "competition_validation_used": False,
        "final_test_used": False,
        "primary_metric": "macro_auroc",
        "metric_constraints": {
            "macro_balanced_accuracy_max_regression": 0.002,
            "macro_brier_max_regression": 0.002,
        },
        "stop_condition": (
            "run the 16 frozen cells once; do not add cells after viewing "
            "internal-development results"
        ),
        "training_recipe": "qlora-nf4-double-quant-bf16-fla-0.5.2",
        "direct_logits_mode": "selected_positions",
        "micro_batch_size": QLORA_MICRO_BATCH_SIZE,
        "gradient_accumulation_steps": QLORA_GRADIENT_ACCUMULATION_STEPS,
        "effective_batch_size": QLORA_EFFECTIVE_BATCH_SIZE,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"prepared {len(jobs)} jobs with {len(training)} training and "
        f"{len(development)} development rows; manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()
