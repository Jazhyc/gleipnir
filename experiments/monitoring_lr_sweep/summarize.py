#!/usr/bin/env python3
"""Summarize the frozen ID screen and apply its replacement gate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from experiments.monitoring_lr_sweep.core import (
    select_screen_winner,
    validate_jobs,
    validate_training_metadata,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def result_row(job: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    if (
        result.get("job_name") != job["job_name"]
        or int(result.get("rows", -1)) != 3_012
    ):
        raise ValueError(f"incomplete or mismatched result for {job['job_name']}")
    metrics = result["metrics"]["macro"]
    macro = metrics["macro"]
    groups = metrics["groups"]
    if {str(group["group"]) for group in groups} != {
        "test_stride",
        "gloom_exfiltration",
    }:
        raise ValueError(f"ID source membership drift for {job['job_name']}")
    validate_training_metadata(
        Path(job["causal_adapter_dir"]) / "training_metadata.json",
        float(job["learning_rate"]),
    )
    return {
        "job_name": job["job_name"],
        "learning_rate": float(job["learning_rate"]),
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
        "training_metadata_sha256": result["training_metadata_sha256"],
        "adapter_rebase_source_sha256": result["adapter_rebase_source_sha256"],
        "adapter_rebase_destination_sha256": result[
            "adapter_rebase_destination_sha256"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jobs",
        type=Path,
        default=Path("results/monitoring_lr_sweep/lambda_jobs.jsonl"),
    )
    parser.add_argument(
        "--runs", type=Path, default=Path("results/monitoring_lr_sweep/id_evaluation")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/monitoring_lr_sweep/summary")
    )
    args = parser.parse_args()
    jobs = read_jsonl(args.jobs.resolve())
    validate_jobs(jobs)
    rows = []
    for job in jobs:
        result_path = (
            args.runs.resolve()
            / "4b"
            / "adapters"
            / str(job["job_name"])
            / "result.json"
        )
        rows.append(result_row(job, json.loads(result_path.read_text())))
    selection = select_screen_winner(rows)
    output = args.output_dir.resolve()
    atomic_write_json(
        output / "summary.json",
        {
            "campaign_id": "gleipnir4b-monitoring-only-lr-sweep-v5",
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
