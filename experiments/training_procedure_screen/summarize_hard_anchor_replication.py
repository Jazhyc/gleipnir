#!/usr/bin/env python3
"""Summarize paired three-seed hard-anchor replication results."""

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

from experiments.adapter_capacity_scaling.prepare import read_jsonl  # noqa: E402
from experiments.training_procedure_screen.summarize import result_row  # noqa: E402

METRICS = ("macro_auroc", "macro_balanced_accuracy", "macro_brier")


def aggregate_replication(combined: pd.DataFrame) -> list[dict[str, Any]]:
    """Aggregate three-seed metrics and paired differences by anchor weight."""
    rows: list[dict[str, Any]] = []
    for weight, group in combined.groupby("direct_loss_weight", sort=True):
        if sorted(group["seed"].astype(int).tolist()) != [0, 1, 2]:
            raise ValueError(f"anchor {weight} does not have exactly seeds 0, 1, 2")
        row: dict[str, Any] = {
            "direct_loss_weight": float(weight),
            "seeds": [0, 1, 2],
        }
        for metric in METRICS:
            row[f"mean_{metric}"] = float(group[metric].mean())
            row[f"std_{metric}"] = float(group[metric].std(ddof=1))
            delta = f"paired_difference_{metric}"
            row[f"mean_{delta}"] = float(group[delta].mean())
            row[f"std_{delta}"] = float(group[delta].std(ddof=1))
        row["passes_mean_metric_constraints"] = bool(
            row["mean_paired_difference_macro_auroc"] > 0
            and row["mean_paired_difference_macro_balanced_accuracy"] >= -0.002
            and row["mean_paired_difference_macro_brier"] <= 0.002
        )
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path("results/training_procedure_screen/hard_anchor_replication")
    parser.add_argument("--jobs", type=Path, default=root / "lambda_jobs.jsonl")
    parser.add_argument(
        "--screen-results",
        type=Path,
        default=Path("results/training_procedure_screen/summary/results.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=root / "summary")
    args = parser.parse_args()
    jobs = read_jsonl(args.jobs.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    replication = pd.DataFrame([result_row(job, "final") for job in jobs])
    replication.to_csv(output_dir / "replication_results.csv", index=False)

    screen = pd.read_csv(args.screen_results.resolve())
    baselines = screen[screen["intervention_family"] == "baseline"]
    seed0 = screen[screen["job_name"].isin(["hard-anchor-010", "hard-anchor-025"])]
    combined = pd.concat([seed0, replication], ignore_index=True)
    baseline_metrics = baselines[["seed", *METRICS]].rename(
        columns={metric: f"baseline_{metric}" for metric in METRICS}
    )
    combined = combined.merge(baseline_metrics, on="seed", validate="many_to_one")
    for metric in METRICS:
        combined[f"paired_difference_{metric}"] = (
            combined[metric] - combined[f"baseline_{metric}"]
        )
    combined.to_csv(output_dir / "paired_results.csv", index=False)
    candidates = aggregate_replication(combined)
    eligible = [row for row in candidates if row["passes_mean_metric_constraints"]]
    preferred = (
        max(eligible, key=lambda row: row["mean_macro_auroc"])[
            "direct_loss_weight"
        ]
        if eligible
        else None
    )
    summary = {
        "primary_metric": "mean paired macro_auroc difference",
        "candidates": candidates,
        "preferred_anchor_weight": preferred,
        "promotion_permitted_within_internal_development": preferred is not None,
        "external_confirmation_required": True,
        "competition_validation_used": False,
        "final_test_used": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
