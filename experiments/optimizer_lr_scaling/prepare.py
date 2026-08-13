#!/usr/bin/env python3
"""Prepare a paired AdamW/Muon learning-rate screen on an internal dev split."""

from __future__ import annotations

import argparse
import json
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
from experiments.data_scaling.core import (  # noqa: E402
    nested_stratified_selections,
)
from experiments.optimizer_lr_scaling.core import (  # noqa: E402
    DEFAULT_LEARNING_RATES,
    DEFAULT_OPTIMIZERS,
    learning_rate_tag,
    validate_learning_rates,
)


def selection_row(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": record["dataset"],
        "index": record["index"],
        "label": int(record["label"]),
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
        "--data-dir",
        type=Path,
        default=Path("data/optimizer_lr_scaling"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/optimizer_lr_scaling"),
    )
    parser.add_argument(
        "--learning-rates",
        nargs="+",
        type=float,
        default=DEFAULT_LEARNING_RATES,
    )
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument("--development-seed", type=int, default=20260813)
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.25)
    parser.add_argument(
        "--development-fraction-of-complement", type=float, default=0.20
    )
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    args = parser.parse_args()

    rates = validate_learning_rates(args.learning_rates)
    if args.rank != 64:
        raise ValueError("the predeclared optimizer screen uses rank 64")
    if not 0 < args.train_fraction < 1:
        raise ValueError("train fraction must be in (0, 1)")
    if not 0 < args.development_fraction_of_complement < 1:
        raise ValueError("development fraction of complement must be in (0, 1)")
    if args.num_train_epochs <= 0:
        raise ValueError("number of training epochs must be positive")

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

    training = nested_stratified_selections(
        records, [args.train_fraction], args.selection_seed
    )[args.train_fraction]
    training_ids = {identity(record) for record in training}
    complement = [record for record in records if identity(record) not in training_ids]
    development = nested_stratified_selections(
        complement,
        [args.development_fraction_of_complement],
        args.development_seed,
    )[args.development_fraction_of_complement]
    development_ids = {identity(record) for record in development}
    if training_ids & development_ids:
        raise AssertionError("optimizer training and development splits overlap")

    selection_path = data_dir / "train_seed0_f025.jsonl"
    development_path = data_dir / "development.jsonl"
    write_jsonl(selection_path, [selection_row(record) for record in training])
    write_jsonl(development_path, development)
    jobs = []
    for learning_rate in rates:
        for optimizer in DEFAULT_OPTIMIZERS:
            job_name = f"{optimizer}-lr{learning_rate_tag(learning_rate)}"
            job_dir = output_dir / "runs" / job_name
            jobs.append(
                {
                    "job_name": job_name,
                    "capacity_kind": "lora",
                    "seed": args.training_seed,
                    "rank": args.rank,
                    "lora_alpha": 2 * args.rank,
                    "optimizer": optimizer,
                    "learning_rate": learning_rate,
                    "lr_scheduler_type": "linear",
                    "warmup_ratio": 0.03,
                    "num_train_epochs": args.num_train_epochs,
                    "weight_decay": 0.0,
                    "muon_adjust_lr_fn": "match_rms_adamw",
                    "train_rows": len(training),
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
                    "gradient_accumulation_steps": (
                        QLORA_GRADIENT_ACCUMULATION_STEPS
                    ),
                    "effective_batch_size": QLORA_EFFECTIVE_BATCH_SIZE,
                }
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs_path = output_dir / "jobs.jsonl"
    write_jsonl(jobs_path, jobs)

    stratum_counts = Counter(
        (str(record["dataset"]), int(record["label"])) for record in development
    )
    manifest = {
        "hypothesis": (
            "RMS-matched Muon improves rank-64 soft-distillation optimization "
            "relative to AdamW at a shared scheduled base learning rate"
        ),
        "intervention": "optimizer at five paired base learning rates",
        "optimizers": list(DEFAULT_OPTIMIZERS),
        "learning_rates": rates,
        "muon_adjust_lr_fn": "match_rms_adamw",
        "lr_scheduler_type": "linear",
        "warmup_ratio": 0.03,
        "num_train_epochs": args.num_train_epochs,
        "rank": args.rank,
        "lora_alpha": 2 * args.rank,
        "training_seed": args.training_seed,
        "selection_seed": args.selection_seed,
        "development_seed": args.development_seed,
        "train_fraction": args.train_fraction,
        "training_rows": len(training),
        "development_fraction_of_complement": (
            args.development_fraction_of_complement
        ),
        "development_rows": len(development),
        "development_strata": len(stratum_counts),
        "lineage_grouping": (
            "unavailable: source rows expose dataset and row identity but no "
            "conversation or generator lineage field"
        ),
        "training_development_overlap": 0,
        "student_rows": student_path.as_posix(),
        "student_rows_sha256": sha256_file(student_path),
        "soft_targets": soft_path.as_posix(),
        "soft_targets_sha256": sha256_file(soft_path),
        "selection_manifest": selection_path.as_posix(),
        "selection_sha256": sha256_file(selection_path),
        "development": development_path.as_posix(),
        "development_sha256": sha256_file(development_path),
        "jobs": len(jobs),
        "jobs_path": jobs_path.as_posix(),
        "competition_validation_used_for_selection": False,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"prepared {len(jobs)} paired optimizer jobs with {len(training)} train "
        f"and {len(development)} internal development rows; manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()
