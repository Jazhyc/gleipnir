#!/usr/bin/env python3
"""Prepare six full-data Muon jobs and validate the reused AdamW controls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

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
from experiments.optimizer_full_data_confirmation.core import (  # noqa: E402
    ADAMW_REFERENCE_LEARNING_RATE,
    CONFIRMATION_LEARNING_RATES,
    CONFIRMATION_SEEDS,
    validate_confirmation_jobs,
)
from experiments.optimizer_lr_scaling.core import learning_rate_tag  # noqa: E402


def validate_adamw_references(path: Path) -> pd.DataFrame:
    """Return the recipe-matched rank-64, full-data, three-seed controls."""
    frame = pd.read_csv(path)
    references = frame[(frame["fraction"] == 1.0) & (frame["rank"] == 64)].copy()
    if set(references["seed"].astype(int)) != set(CONFIRMATION_SEEDS):
        raise ValueError("AdamW references must contain rank-64 seeds 0, 1, and 2")
    if len(references) != 3 or not (references["train_rows"] == 13149).all():
        raise ValueError("AdamW references must be three 13,149-row full-data cells")
    return references.sort_values("seed")


def job_row(
    seed: int,
    learning_rate: float,
    student_path: Path,
    soft_path: Path,
    validation_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    job_name = f"seed{seed}-muon-lr{learning_rate_tag(learning_rate)}"
    job_dir = output_dir / "runs" / job_name
    return {
        "job_name": job_name,
        "capacity_kind": "lora",
        "seed": seed,
        "fraction": 1.0,
        "rank": 64,
        "lora_alpha": 128,
        "optimizer": "muon",
        "learning_rate": learning_rate,
        "lr_scheduler_type": "linear",
        "warmup_ratio": 0.03,
        "num_train_epochs": 2.0,
        "weight_decay": 0.0,
        "muon_adjust_lr_fn": "match_rms_adamw",
        "train_rows": 13149,
        "student_rows": student_path.as_posix(),
        "soft_targets": soft_path.as_posix(),
        "validation": validation_path.as_posix(),
        "output_dir": job_dir.as_posix(),
        "causal_adapter_dir": (job_dir / "causal_adapter").as_posix(),
        "model_dir": (job_dir / "model").as_posix(),
        "training_recipe": "qlora-nf4-double-quant-bf16-fla-0.5.2",
        "micro_batch_size": QLORA_MICRO_BATCH_SIZE,
        "gradient_accumulation_steps": QLORA_GRADIENT_ACCUMULATION_STEPS,
        "effective_batch_size": QLORA_EFFECTIVE_BATCH_SIZE,
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
        "--validation",
        type=Path,
        default=Path("data/data_scaling/validation.jsonl"),
    )
    parser.add_argument(
        "--adamw-references",
        type=Path,
        default=Path("results/data_capacity_scaling_qlora/summary/replicates.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/optimizer_full_data_confirmation"),
    )
    args = parser.parse_args()

    student_path = args.student_rows.resolve()
    soft_path = args.soft_targets.resolve()
    validation_path = args.validation.resolve()
    reference_path = args.adamw_references.resolve()
    output_dir = args.output_dir.resolve()
    records = read_jsonl(student_path)
    targets = read_jsonl(soft_path)
    validation = read_jsonl(validation_path)
    if len(records) != 13149 or len(targets) != len(records):
        raise ValueError("expected matching 13,149-row student and target artifacts")
    if len(validation) != 822:
        raise ValueError("expected the frozen 822-row validation artifact")
    target_by_id = {identity(record): record for record in targets}
    if len(target_by_id) != len(targets):
        raise ValueError("soft-target artifact contains duplicate identities")
    for record in records:
        target = target_by_id.get(identity(record))
        if target is None or int(target["label"]) != int(record["label"]):
            raise ValueError(f"missing or inconsistent target for {identity(record)!r}")
    overlap = {identity(record) for record in records} & {
        identity(record) for record in validation
    }
    if overlap:
        raise ValueError(f"training/validation identity leakage: {sorted(overlap)[:3]}")
    references = validate_adamw_references(reference_path)

    jobs = [
        job_row(
            seed,
            learning_rate,
            student_path,
            soft_path,
            validation_path,
            output_dir,
        )
        for seed in CONFIRMATION_SEEDS
        for learning_rate in CONFIRMATION_LEARNING_RATES
    ]
    validate_confirmation_jobs(jobs)
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs_path = output_dir / "jobs.jsonl"
    reference_export = output_dir / "adamw_references.csv"
    write_jsonl(jobs_path, jobs)
    references.to_csv(reference_export, index=False)
    manifest = {
        "hypothesis": (
            "the two Muon rates advanced by the internal screen improve "
            "full-data rank-64 generalization over recipe-matched AdamW"
        ),
        "intervention": "RMS-matched Muon learning rate at 5e-5 or 1e-4",
        "muon_learning_rates": list(CONFIRMATION_LEARNING_RATES),
        "adamw_reference_learning_rate": ADAMW_REFERENCE_LEARNING_RATE,
        "seeds": list(CONFIRMATION_SEEDS),
        "new_muon_jobs": len(jobs),
        "reused_adamw_jobs": len(references),
        "rank": 64,
        "lora_alpha": 128,
        "training_rows": len(records),
        "validation_rows": len(validation),
        "train_validation_overlap": 0,
        "student_rows": student_path.as_posix(),
        "student_rows_sha256": sha256_file(student_path),
        "soft_targets": soft_path.as_posix(),
        "soft_targets_sha256": sha256_file(soft_path),
        "validation": validation_path.as_posix(),
        "validation_sha256": sha256_file(validation_path),
        "adamw_references": reference_path.as_posix(),
        "adamw_references_sha256": sha256_file(reference_path),
        "adamw_reference_export": reference_export.as_posix(),
        "adamw_reference_export_sha256": sha256_file(reference_export),
        "adamw_reference_source_commit": (
            "f9dcc365be6b67ab0bac7abe7d70161f9e36957a"
        ),
        "lr_scheduler_type": "linear",
        "warmup_ratio": 0.03,
        "num_train_epochs": 2.0,
        "muon_adjust_lr_fn": "match_rms_adamw",
        "training_recipe": "qlora-nf4-double-quant-bf16-fla-0.5.2",
        "direct_logits_mode": "selected_positions",
        "micro_batch_size": QLORA_MICRO_BATCH_SIZE,
        "gradient_accumulation_steps": QLORA_GRADIENT_ACCUMULATION_STEPS,
        "effective_batch_size": QLORA_EFFECTIVE_BATCH_SIZE,
        "selection_rule": (
            "choose the eligible optimizer/rate with highest mean macro AUROC; "
            "Muon must improve paired mean AUROC in at least two of three seeds, "
            "with mean macro BA regression <=0.002 and Brier regression <=0.002"
        ),
        "competition_validation_use": (
            "confirmatory evaluation of two predeclared Muon candidates; do not "
            "add optimizer rates or retune on these results"
        ),
        "final_test_used": False,
        "jobs_path": jobs_path.as_posix(),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"prepared {len(jobs)} full-data Muon jobs and validated "
        f"{len(references)} AdamW references; manifest={output_dir / 'manifest.json'}"
    )


if __name__ == "__main__":
    main()
