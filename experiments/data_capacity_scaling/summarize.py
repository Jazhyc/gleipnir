#!/usr/bin/env python3
"""Summarize the joint data-volume and QLoRA-rank scaling surface."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.adapter_capacity_scaling.prepare import read_jsonl  # noqa: E402
from experiments.adapter_capacity_scaling.summarize import (  # noqa: E402
    macro_metric,
    safe_fit,
)
from experiments.data_capacity_scaling.core import DEFAULT_RANKS  # noqa: E402


def result_row(job: dict[str, Any], fraction: float) -> dict[str, Any]:
    result_path = Path(job["output_dir"]) / "validation" / "result.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"missing evaluation result: {result_path}")
    result = json.loads(result_path.read_text())
    diagnostics = result["metrics"]["macro"]["diagnostics"]
    return {
        "job_name": job["job_name"],
        "seed": int(job["seed"]),
        "fraction": fraction,
        "train_rows": int(job["train_rows"]),
        "rank": int(job["rank"]),
        "lora_alpha": int(job["lora_alpha"]),
        "trainable_parameters": int(result["trainable_parameters"]),
        "macro_auroc": macro_metric(result, "auroc"),
        "macro_balanced_accuracy": macro_metric(result, "balanced_accuracy"),
        "macro_brier": macro_metric(result, "brier"),
        "pooled_auroc": float(result["metrics"]["pooled"]["auroc"]),
        "pooled_balanced_accuracy": float(
            result["metrics"]["pooled"]["balanced_accuracy"]
        ),
        "pooled_brier": float(result["metrics"]["pooled"]["brier"]),
        "unique_scores": int(diagnostics["unique_scores"]),
        "tied_rows": int(diagnostics["tied_rows"]),
    }


def collect_rows(
    jobs: list[dict[str, Any]], full_data_jobs: list[dict[str, Any]]
) -> pd.DataFrame:
    rows = [result_row(job, float(job["fraction"])) for job in jobs]
    references = [
        job
        for job in full_data_jobs
        if job["capacity_kind"] == "lora" and int(job["rank"]) in DEFAULT_RANKS
    ]
    rows.extend(result_row(job, 1.0) for job in references)
    frame = pd.DataFrame(rows).sort_values(["fraction", "rank", "seed"])
    expected = {
        (fraction, rank, seed)
        for fraction in (0.10, 0.25, 0.50, 1.0)
        for rank in DEFAULT_RANKS
        for seed in (0, 1, 2)
    }
    observed = {
        (round(float(row.fraction), 2), int(row.rank), int(row.seed))
        for row in frame.itertuples()
    }
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"incomplete interaction grid: missing={missing[:3]} extra={extra[:3]}"
        )
    return frame


def paired_contrasts(frame: pd.DataFrame, baseline_rank: int = 16) -> pd.DataFrame:
    wide = frame.pivot(index=["fraction", "train_rows", "seed"], columns="rank")
    rows = []
    for rank in DEFAULT_RANKS:
        if rank == baseline_rank:
            continue
        for metric in (
            "macro_auroc",
            "macro_balanced_accuracy",
            "macro_brier",
            "pooled_auroc",
        ):
            difference = wide[metric][rank] - wide[metric][baseline_rank]
            for (fraction, train_rows), values in difference.groupby(
                level=["fraction", "train_rows"]
            ):
                rows.append(
                    {
                        "fraction": fraction,
                        "train_rows": train_rows,
                        "rank": rank,
                        "baseline_rank": baseline_rank,
                        "metric": metric,
                        "paired_mean_difference": float(values.mean()),
                        "paired_std": float(values.std()),
                        "seeds": int(values.count()),
                    }
                )
    return pd.DataFrame(rows).sort_values(["metric", "fraction", "rank"])


def high_rank_interaction(frame: pd.DataFrame, high_rank: int = 256) -> dict[str, Any]:
    """Regress paired rank-256 minus rank-16 AUROC on log training rows."""
    wide = frame.pivot(index=["fraction", "train_rows", "seed"], columns="rank")
    differences = (
        wide["macro_auroc"][high_rank] - wide["macro_auroc"][16]
    ).reset_index(name="difference")
    log_rows = np.log(differences["train_rows"].to_numpy(dtype=float))
    seeds = differences["seed"].to_numpy(dtype=int)
    design = np.column_stack(
        [
            np.ones(len(differences)),
            log_rows - log_rows.mean(),
            seeds == 1,
            seeds == 2,
        ]
    ).astype(float)
    target = differences["difference"].to_numpy(dtype=float)
    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ coefficients
    degrees_of_freedom = len(target) - design.shape[1]
    residual_sd = math.sqrt(float(residual @ residual) / degrees_of_freedom)
    covariance = residual_sd**2 * np.linalg.inv(design.T @ design)
    slope_se = math.sqrt(float(covariance[1, 1]))
    return {
        "contrast": f"rank_{high_rank}_minus_rank_16_macro_auroc",
        "model": "paired_difference ~ centered_log(train_rows) + seed_fixed_effects",
        "log_train_rows_slope": float(coefficients[1]),
        "log_train_rows_slope_standard_error": slope_se,
        "residual_sd": residual_sd,
        "observations": len(target),
        "interpretation": (
            "positive means the high-rank advantage increases with data volume"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jobs",
        type=Path,
        default=Path("results/data_capacity_scaling_qlora/lambda_jobs.jsonl"),
    )
    parser.add_argument(
        "--full-data-jobs",
        type=Path,
        default=Path("results/adapter_capacity_scaling_qlora/lambda_jobs.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    jobs_path = args.jobs.resolve()
    output_dir = (args.output_dir or jobs_path.parent / "summary").resolve()
    frame = collect_rows(
        read_jsonl(jobs_path), read_jsonl(args.full_data_jobs.resolve())
    )
    grouped = (
        frame.groupby(["fraction", "train_rows", "rank"], sort=True)
        .agg(
            seeds=("seed", "count"),
            trainable_parameters=("trainable_parameters", "mean"),
            macro_auroc_mean=("macro_auroc", "mean"),
            macro_auroc_std=("macro_auroc", "std"),
            macro_balanced_accuracy_mean=("macro_balanced_accuracy", "mean"),
            macro_balanced_accuracy_std=("macro_balanced_accuracy", "std"),
            macro_brier_mean=("macro_brier", "mean"),
            macro_brier_std=("macro_brier", "std"),
            pooled_auroc_mean=("pooled_auroc", "mean"),
            pooled_auroc_std=("pooled_auroc", "std"),
            pooled_balanced_accuracy_mean=("pooled_balanced_accuracy", "mean"),
            pooled_balanced_accuracy_std=("pooled_balanced_accuracy", "std"),
            pooled_brier_mean=("pooled_brier", "mean"),
            pooled_brier_std=("pooled_brier", "std"),
        )
        .reset_index()
    )
    contrasts = paired_contrasts(frame)
    fits: dict[str, Any] = {}
    best_by_fraction: dict[str, Any] = {}
    for fraction, points in grouped.groupby("fraction", sort=True):
        key = f"{fraction:.2f}"
        fits[key] = {}
        for metric in (
            "macro_auroc",
            "macro_balanced_accuracy",
            "pooled_auroc",
            "pooled_balanced_accuracy",
        ):
            fits[key][metric] = safe_fit(points["rank"], points[f"{metric}_mean"])
        best = points.loc[points["macro_auroc_mean"].idxmax()]
        best_by_fraction[key] = {
            "rank": int(best["rank"]),
            "train_rows": int(best["train_rows"]),
            "macro_auroc_mean": float(best["macro_auroc_mean"]),
            "macro_auroc_std": float(best["macro_auroc_std"]),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "replicates.csv", index=False)
    grouped.to_csv(output_dir / "surface.csv", index=False)
    contrasts.to_csv(output_dir / "paired_contrasts.csv", index=False)
    summary = {
        "primary_metric": "macro_auroc",
        "fractions": [0.10, 0.25, 0.50, 1.0],
        "ranks": list(DEFAULT_RANKS),
        "seeds": [0, 1, 2],
        "new_training_jobs": len(frame[frame["fraction"] < 1.0]),
        "reused_full_data_results": len(frame[frame["fraction"] == 1.0]),
        "best_observed_rank_by_fraction": best_by_fraction,
        "rank_curve_fits": fits,
        "high_rank_interaction": high_rank_interaction(frame),
    }
    (output_dir / "interaction_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(grouped.to_string(index=False))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
