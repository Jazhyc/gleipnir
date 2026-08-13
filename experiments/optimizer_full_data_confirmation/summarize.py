#!/usr/bin/env python3
"""Summarize full-data Muon confirmation against reused AdamW controls."""

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
from experiments.adapter_capacity_scaling.summarize import macro_metric  # noqa: E402
from experiments.optimizer_full_data_confirmation.core import (  # noqa: E402
    ADAMW_REFERENCE_LEARNING_RATE,
    CONFIRMATION_LEARNING_RATES,
    CONFIRMATION_SEEDS,
)

METRICS = [
    "macro_auroc",
    "macro_balanced_accuracy",
    "macro_brier",
    "pooled_auroc",
    "pooled_balanced_accuracy",
    "pooled_brier",
]


def muon_result_row(job: dict[str, Any]) -> dict[str, Any]:
    result_path = Path(job["output_dir"]) / "validation" / "result.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"missing confirmation result: {result_path}")
    result = json.loads(result_path.read_text())
    diagnostics = result["metrics"]["macro"]["diagnostics"]
    training = json.loads(
        (Path(job["model_dir"]) / "training_metadata.json").read_text()
    )
    train_metrics = training["train_metrics"]
    return {
        "job_name": job["job_name"],
        "optimizer": "muon",
        "learning_rate": float(job["learning_rate"]),
        "seed": int(job["seed"]),
        "train_rows": int(job["train_rows"]),
        "rank": int(job["rank"]),
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
        "train_loss": float(train_metrics["train_loss"]),
        "train_runtime_seconds": float(train_metrics["train_runtime"]),
        "train_samples_per_second": float(
            train_metrics["train_samples_per_second"]
        ),
    }


def adamw_reference_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    selected = frame[(frame["fraction"] == 1.0) & (frame["rank"] == 64)].copy()
    if len(selected) != 3 or set(selected["seed"].astype(int)) != set(
        CONFIRMATION_SEEDS
    ):
        raise ValueError("expected three full-data rank-64 AdamW references")
    selected["optimizer"] = "adamw"
    selected["learning_rate"] = ADAMW_REFERENCE_LEARNING_RATE
    return selected


def paired_confirmation_contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    baseline = frame[frame["optimizer"] == "adamw"].set_index("seed")
    rows = []
    for learning_rate in CONFIRMATION_LEARNING_RATES:
        candidate = frame[
            (frame["optimizer"] == "muon")
            & (frame["learning_rate"] == learning_rate)
        ].set_index("seed")
        if set(candidate.index) != set(CONFIRMATION_SEEDS):
            raise ValueError(f"incomplete Muon seed set at {learning_rate}")
        for seed in CONFIRMATION_SEEDS:
            row: dict[str, Any] = {
                "seed": seed,
                "muon_learning_rate": learning_rate,
            }
            for metric in METRICS:
                row[f"muon_minus_adamw_{metric}"] = float(
                    candidate.loc[seed, metric] - baseline.loc[seed, metric]
                )
            rows.append(row)
    return pd.DataFrame(rows)


def aggregate_results(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["optimizer", "learning_rate"], sort=True)
        .agg(
            seeds=("seed", "count"),
            macro_auroc_mean=("macro_auroc", "mean"),
            macro_auroc_std=("macro_auroc", "std"),
            macro_balanced_accuracy_mean=("macro_balanced_accuracy", "mean"),
            macro_balanced_accuracy_std=("macro_balanced_accuracy", "std"),
            macro_brier_mean=("macro_brier", "mean"),
            macro_brier_std=("macro_brier", "std"),
            pooled_auroc_mean=("pooled_auroc", "mean"),
            pooled_auroc_std=("pooled_auroc", "std"),
            unique_scores_mean=("unique_scores", "mean"),
            tied_rows_mean=("tied_rows", "mean"),
        )
        .reset_index()
    )


def selection_summary(
    aggregate: pd.DataFrame, contrasts: pd.DataFrame
) -> dict[str, Any]:
    adamw = aggregate[aggregate["optimizer"] == "adamw"].iloc[0]
    candidates = []
    for row in aggregate[aggregate["optimizer"] == "muon"].itertuples():
        paired = contrasts[contrasts["muon_learning_rate"] == row.learning_rate]
        auroc_difference = paired["muon_minus_adamw_macro_auroc"]
        ba_difference = float(
            paired["muon_minus_adamw_macro_balanced_accuracy"].mean()
        )
        brier_difference = float(paired["muon_minus_adamw_macro_brier"].mean())
        eligible = (
            float(auroc_difference.mean()) > 0
            and int((auroc_difference > 0).sum()) >= 2
            and ba_difference >= -0.002
            and brier_difference <= 0.002
        )
        candidates.append(
            {
                "optimizer": "muon",
                "learning_rate": float(row.learning_rate),
                "macro_auroc_mean": float(row.macro_auroc_mean),
                "paired_macro_auroc_mean_difference": float(
                    auroc_difference.mean()
                ),
                "seeds_with_positive_auroc_difference": int(
                    (auroc_difference > 0).sum()
                ),
                "paired_macro_balanced_accuracy_mean_difference": ba_difference,
                "paired_macro_brier_mean_difference": brier_difference,
                "eligible": eligible,
            }
        )
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    selected = (
        max(eligible, key=lambda candidate: candidate["macro_auroc_mean"])
        if eligible
        else {
            "optimizer": "adamw",
            "learning_rate": ADAMW_REFERENCE_LEARNING_RATE,
            "macro_auroc_mean": float(adamw["macro_auroc_mean"]),
            "reason": "no Muon candidate passed the preregistered gate",
        }
    )
    return {
        "primary_metric": "macro_auroc",
        "selection_rule": (
            "highest mean macro AUROC among Muon candidates with positive paired "
            "mean AUROC in at least two seeds, mean BA regression <=0.002, and "
            "mean Brier regression <=0.002; otherwise retain AdamW"
        ),
        "adamw_reference": {
            "learning_rate": ADAMW_REFERENCE_LEARNING_RATE,
            "macro_auroc_mean": float(adamw["macro_auroc_mean"]),
        },
        "muon_candidates": candidates,
        "selected_recipe": selected,
        "competition_validation_use": "confirmatory",
        "final_test_used": False,
        "further_optimizer_tuning_on_validation_permitted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jobs",
        type=Path,
        default=Path("results/optimizer_full_data_confirmation/lambda_jobs.jsonl"),
    )
    parser.add_argument(
        "--adamw-references",
        type=Path,
        default=Path(
            "results/optimizer_full_data_confirmation/adamw_references.csv"
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    jobs_path = args.jobs.resolve()
    output_dir = (args.output_dir or jobs_path.parent / "summary").resolve()
    muon = pd.DataFrame(muon_result_row(job) for job in read_jsonl(jobs_path))
    adamw = adamw_reference_rows(args.adamw_references.resolve())
    common_columns = sorted(set(muon.columns) & set(adamw.columns))
    frame = pd.concat(
        [adamw[common_columns], muon[common_columns]], ignore_index=True
    ).sort_values(["optimizer", "learning_rate", "seed"])
    contrasts = paired_confirmation_contrasts(frame)
    aggregate = aggregate_results(frame)
    summary = selection_summary(aggregate, contrasts)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "replicates.csv", index=False)
    aggregate.to_csv(output_dir / "aggregate.csv", index=False)
    contrasts.to_csv(output_dir / "paired_contrasts.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(aggregate.to_string(index=False))
    print(contrasts.to_string(index=False))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
