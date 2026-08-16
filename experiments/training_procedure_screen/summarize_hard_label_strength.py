#!/usr/bin/env python3
"""Summarize the predeclared three-seed hard-label strength curve."""

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


def aggregate_strength_curve(combined: pd.DataFrame) -> list[dict[str, Any]]:
    """Return per-strength three-seed means and paired uncertainty."""
    output: list[dict[str, Any]] = []
    for strength_id, group in combined.groupby("strength_id", sort=False):
        if sorted(group["seed"].astype(int).tolist()) != [0, 1, 2]:
            raise ValueError(f"strength {strength_id} lacks seeds 0, 1, and 2")
        first = group.iloc[0]
        row: dict[str, Any] = {
            "strength_id": str(strength_id),
            "soft_loss_weight": float(first["soft_loss_weight"]),
            "direct_loss_weight": float(first["direct_loss_weight"]),
            "hard_label_fraction": float(first["hard_label_fraction"]),
            "seeds": [0, 1, 2],
        }
        for metric in METRICS:
            row[f"mean_{metric}"] = float(group[metric].mean())
            row[f"std_{metric}"] = float(group[metric].std(ddof=1))
            difference = f"paired_difference_{metric}"
            row[f"mean_{difference}"] = float(group[difference].mean())
            row[f"std_{difference}"] = float(group[difference].std(ddof=1))
        row["passes_mean_metric_constraints"] = bool(
            row["mean_paired_difference_macro_auroc"] > 0
            and row["mean_paired_difference_macro_balanced_accuracy"] >= -0.002
            and row["mean_paired_difference_macro_brier"] <= 0.002
        )
        output.append(row)
    return sorted(output, key=lambda row: row["hard_label_fraction"])


def tagged(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if "soft_loss_weight" not in output:
        output["soft_loss_weight"] = 1.0
    output["strength_id"] = output.apply(
        lambda row: (
            "hard-only"
            if float(row["soft_loss_weight"]) == 0.0
            else f"weight-{float(row['direct_loss_weight']):g}"
        ),
        axis=1,
    )
    denominator = output["soft_loss_weight"] + output["direct_loss_weight"]
    output["hard_label_fraction"] = output["direct_loss_weight"] / denominator
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path("results/training_procedure_screen/hard_label_strength")
    parser.add_argument("--jobs", type=Path, default=root / "lambda_jobs.jsonl")
    parser.add_argument(
        "--screen-results",
        type=Path,
        default=Path("results/training_procedure_screen/summary/results.csv"),
    )
    parser.add_argument(
        "--replication-results",
        type=Path,
        default=Path(
            "results/training_procedure_screen/hard_anchor_replication/summary/"
            "replication_results.csv"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=root / "summary")
    args = parser.parse_args()
    jobs = read_jsonl(args.jobs.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    new_results = pd.DataFrame([result_row(job, "final") for job in jobs])
    new_results = new_results.merge(
        pd.DataFrame(
            [
                {
                    "job_name": job["job_name"],
                    "strength_id": job["strength_id"],
                    "hard_label_fraction": job["hard_label_fraction"],
                    "soft_loss_weight": job["soft_loss_weight"],
                }
                for job in jobs
            ]
        ),
        on="job_name",
        validate="one_to_one",
    )
    new_results.to_csv(output_dir / "new_results.csv", index=False)

    screen = pd.read_csv(args.screen_results.resolve())
    baselines = screen[screen["intervention_family"] == "baseline"].copy()
    baselines["soft_loss_weight"] = 1.0
    baselines["strength_id"] = "weight-0"
    baselines["hard_label_fraction"] = 0.0
    seed0 = tagged(
        screen[screen["job_name"].isin(["hard-anchor-010", "hard-anchor-025"])]
    )
    replication = tagged(pd.read_csv(args.replication_results.resolve()))
    combined = pd.concat(
        [baselines, seed0, replication, new_results], ignore_index=True
    )
    if len(combined) != 42:
        raise ValueError(f"expected 14 strengths x 3 seeds, got {len(combined)} rows")
    baseline_metrics = baselines[["seed", *METRICS]].rename(
        columns={metric: f"baseline_{metric}" for metric in METRICS}
    )
    combined = combined.merge(baseline_metrics, on="seed", validate="many_to_one")
    for metric in METRICS:
        combined[f"paired_difference_{metric}"] = (
            combined[metric] - combined[f"baseline_{metric}"]
        )
    combined.to_csv(output_dir / "paired_curve_results.csv", index=False)
    curve = aggregate_strength_curve(combined)
    pd.DataFrame(curve).to_csv(output_dir / "strength_curve.csv", index=False)
    eligible = [row for row in curve if row["passes_mean_metric_constraints"]]
    preferred = (
        max(eligible, key=lambda row: row["mean_macro_auroc"])["strength_id"]
        if eligible
        else None
    )
    summary = {
        "primary_metric": "three-seed mean paired macro_auroc difference",
        "curve": curve,
        "preferred_strength_id": preferred,
        "selection_candidates": len(curve),
        "adaptive_grid_extension_permitted": False,
        "external_confirmation_required": True,
        "competition_validation_used": False,
        "final_test_used": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
