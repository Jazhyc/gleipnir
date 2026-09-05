#!/usr/bin/env python3
"""Summarize the frozen ID objective-ablation screen."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from experiments.monitoring_objective_ablation.core import (
    CAMPAIGN_ID,
    CONTROL,
    select_winner,
    validate_jobs,
    validate_training_metadata,
)
from gleipnir.monitoring_systems_screen import atomic_write_json, read_jsonl


def result_row(job: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    if result.get("job_name") != job["job_name"] or int(result.get("rows", -1)) != 3012:
        raise ValueError(f"incomplete result for {job['job_name']}")
    training = validate_training_metadata(
        Path(job["causal_adapter_dir"]) / "training_metadata.json", job
    )
    metrics = result["metrics"]["macro"]
    macro = metrics["macro"]
    groups = metrics["groups"]
    return {
        "job_name": str(job["job_name"]),
        "objective_accumulation_valid": (
            not bool(job["completion_loss_weight"])
            or training.get("losses", {}).get("accumulation_policy")
            == "explicit_microbatch_mean_v1"
        ),
        "kind": str(job["kind"]),
        "completion_loss_weight": float(job["completion_loss_weight"]),
        "mil_loss_weight": float(job["mil_loss_weight"]),
        "mil_pooling": str(job["mil_pooling"]),
        "macro_pauroc_at_20": float(macro["pauroc_at_20"]),
        "macro_auroc": float(macro["auroc"]),
        "macro_brier": float(macro["brier"]),
        "macro_balanced_accuracy": float(macro["balanced_accuracy"]),
        "macro_recall": float(macro["recall"]),
        "macro_fpr": float(macro["fpr"]),
        "source_pauroc_at_20": {
            str(group["group"]): float(group["pauroc_at_20"]) for group in groups
        },
        "unique_scores": int(metrics["diagnostics"]["unique_scores"]),
        "tied_rows": int(metrics["diagnostics"]["tied_rows"]),
        "elapsed_seconds": float(result["runtime"]["elapsed_seconds_this_invocation"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jobs",
        type=Path,
        default=Path("results/monitoring_objective_ablation/jobs.jsonl"),
    )
    parser.add_argument(
        "--runs",
        type=Path,
        default=Path("results/monitoring_objective_ablation/id_evaluation"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/monitoring_objective_ablation/summary"),
    )
    args = parser.parse_args()
    jobs = read_jsonl(args.jobs.resolve())
    validate_jobs(jobs)
    rows = []
    for job in jobs:
        result = json.loads(
            (
                args.runs.resolve()
                / "4b"
                / "adapters"
                / str(job["job_name"])
                / "result.json"
            ).read_text()
        )
        rows.append(result_row(job, result))
    selection = select_winner(rows)
    output = args.output_dir.resolve()
    atomic_write_json(
        output / "summary.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "control": CONTROL,
            "rows": rows,
            "selection": selection,
            "strict_ood_consulted": False,
            "interpretation_scope": "one-seed ID-development pruning",
        },
    )
    output.mkdir(parents=True, exist_ok=True)
    flat_rows = []
    for row in rows:
        flat = {key: value for key, value in row.items() if not isinstance(value, dict)}
        flat.update(
            {
                f"source_pauroc_at_20_{source}": score
                for source, score in row["source_pauroc_at_20"].items()
            }
        )
        flat_rows.append(flat)
    with (output / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    print(json.dumps(selection, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
