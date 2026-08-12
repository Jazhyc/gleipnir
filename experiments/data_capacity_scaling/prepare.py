#!/usr/bin/env python3
"""Prepare the missing cells of the data-volume by QLoRA-rank grid."""

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
from experiments.data_capacity_scaling.core import (  # noqa: E402
    DEFAULT_FRACTIONS,
    DEFAULT_RANKS,
    DEFAULT_SEEDS,
    fraction_tag,
    validate_design,
)
from experiments.data_scaling.core import (  # noqa: E402
    nested_stratified_selections,
)


def manifest_row(record: dict[str, Any]) -> dict[str, Any]:
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
        "--validation",
        type=Path,
        default=Path("data/data_scaling/validation.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/data_capacity_scaling_qlora"),
    )
    parser.add_argument("--fractions", nargs="+", type=float, default=DEFAULT_FRACTIONS)
    parser.add_argument("--ranks", nargs="+", type=int, default=DEFAULT_RANKS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    args = parser.parse_args()

    fractions, ranks, seeds = validate_design(args.fractions, args.ranks, args.seeds)
    student_path = args.student_rows.resolve()
    soft_path = args.soft_targets.resolve()
    validation_path = args.validation.resolve()
    output_dir = args.output_dir.resolve()
    records = read_jsonl(student_path)
    targets = read_jsonl(soft_path)
    validation = read_jsonl(validation_path)
    if len(records) != 13149 or len(targets) != len(records):
        raise ValueError("expected matching 13,149-row student and target artifacts")
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

    jobs: list[dict[str, Any]] = []
    selection_summary: dict[str, Any] = {}
    for seed in seeds:
        selections = nested_stratified_selections(records, fractions, seed)
        previous: set[tuple[str, Any]] = set()
        seed_summary: dict[str, Any] = {}
        for fraction, selected in selections.items():
            tag = fraction_tag(fraction)
            selected_ids = {identity(record) for record in selected}
            if not previous.issubset(selected_ids):
                raise AssertionError("prepared selections are not nested")
            previous = selected_ids
            selection_path = output_dir / "selections" / f"seed{seed}-{tag}.jsonl"
            write_jsonl(selection_path, [manifest_row(record) for record in selected])
            counts = Counter(
                (str(record["dataset"]), int(record["label"])) for record in selected
            )
            seed_summary[tag] = {
                "fraction": fraction,
                "rows": len(selected),
                "selection_sha256": sha256_file(selection_path),
                "strata": len(counts),
            }
            for rank in ranks:
                job_name = f"seed{seed}-{tag}-r{rank:03d}"
                job_dir = output_dir / "runs" / job_name
                jobs.append(
                    {
                        "job_name": job_name,
                        "capacity_kind": "lora",
                        "seed": seed,
                        "fraction": fraction,
                        "rank": rank,
                        "lora_alpha": 2 * rank,
                        "train_rows": len(selected),
                        "selection_manifest": selection_path.as_posix(),
                        "selection_sha256": sha256_file(selection_path),
                        "student_rows": student_path.as_posix(),
                        "soft_targets": soft_path.as_posix(),
                        "validation": validation_path.as_posix(),
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
        selection_summary[str(seed)] = seed_summary

    jobs_path = output_dir / "jobs.jsonl"
    write_jsonl(jobs_path, jobs)
    reference_jobs = Path(
        "results/adapter_capacity_scaling_qlora/lambda_jobs.jsonl"
    ).resolve()
    manifest = {
        "hypothesis": (
            "the QLoRA rank needed to saturate held-out monitor AUROC increases "
            "with the number of annotated training examples"
        ),
        "intervention": "crossed nested training-data fraction and QLoRA rank",
        "fractions_trained": fractions,
        "full_data_reference_fraction": 1.0,
        "ranks": ranks,
        "seeds": seeds,
        "new_jobs": len(jobs),
        "referenced_full_data_jobs": len(ranks) * len(seeds),
        "jobs_path": jobs_path.as_posix(),
        "full_data_reference_jobs": reference_jobs.as_posix(),
        "student_rows": student_path.as_posix(),
        "student_rows_sha256": sha256_file(student_path),
        "soft_targets": soft_path.as_posix(),
        "soft_targets_sha256": sha256_file(soft_path),
        "validation": validation_path.as_posix(),
        "validation_sha256": sha256_file(validation_path),
        "training_rows": len(records),
        "validation_rows": len(validation),
        "train_validation_overlap": 0,
        "selection": selection_summary,
        "training_recipe": "qlora-nf4-double-quant-bf16-fla-0.5.2",
        "direct_logits_mode": "selected_positions",
        "micro_batch_size": QLORA_MICRO_BATCH_SIZE,
        "gradient_accumulation_steps": QLORA_GRADIENT_ACCUMULATION_STEPS,
        "effective_batch_size": QLORA_EFFECTIVE_BATCH_SIZE,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"prepared {len(jobs)} new jobs and {len(ranks) * len(seeds)} full-data "
        f"references; manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()
