#!/usr/bin/env python3
"""Summarize paired AdamW and RMS-matched Muon learning-rate results."""

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

CONTRAST_METRICS = [
    "macro_auroc",
    "macro_balanced_accuracy",
    "macro_brier",
    "pooled_auroc",
    "pooled_balanced_accuracy",
    "pooled_brier",
    "train_loss",
    "train_runtime_seconds",
    "train_samples_per_second",
]


def result_row(job: dict[str, Any]) -> dict[str, Any]:
    result_path = Path(job["output_dir"]) / "validation" / "result.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"missing internal development result: {result_path}")
    result = json.loads(result_path.read_text())
    training = json.loads(
        (Path(job["model_dir"]) / "training_metadata.json").read_text()
    )
    metrics = training["train_metrics"]
    return {
        "job_name": job["job_name"],
        "optimizer": job["optimizer"],
        "learning_rate": float(job["learning_rate"]),
        "lr_scheduler_type": job["lr_scheduler_type"],
        "muon_adjust_lr_fn": (
            job["muon_adjust_lr_fn"] if job["optimizer"] == "muon" else None
        ),
        "rank": int(job["rank"]),
        "train_rows": int(job["train_rows"]),
        "macro_auroc": macro_metric(result, "auroc"),
        "macro_balanced_accuracy": macro_metric(result, "balanced_accuracy"),
        "macro_brier": macro_metric(result, "brier"),
        "pooled_auroc": float(result["metrics"]["pooled"]["auroc"]),
        "pooled_balanced_accuracy": float(
            result["metrics"]["pooled"]["balanced_accuracy"]
        ),
        "pooled_brier": float(result["metrics"]["pooled"]["brier"]),
        "train_loss": float(metrics["train_loss"]),
        "train_runtime_seconds": float(metrics["train_runtime"]),
        "train_samples_per_second": float(metrics["train_samples_per_second"]),
        "peak_cuda_memory_allocated_bytes": training[
            "peak_cuda_memory_allocated_bytes"
        ],
    }


def paired_contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute Muon-minus-AdamW contrasts for every paired learning rate."""
    wide = frame.pivot(
        index="learning_rate",
        columns="optimizer",
        values=CONTRAST_METRICS,
    )
    contrasts = []
    for learning_rate in sorted(frame["learning_rate"].unique()):
        row: dict[str, Any] = {"learning_rate": learning_rate}
        for metric in CONTRAST_METRICS:
            row[f"muon_minus_adamw_{metric}"] = float(
                wide.loc[learning_rate, (metric, "muon")]
                - wide.loc[learning_rate, (metric, "adamw")]
            )
        contrasts.append(row)
    return pd.DataFrame(contrasts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jobs",
        type=Path,
        default=Path("results/optimizer_lr_scaling/lambda_jobs.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    jobs_path = args.jobs.resolve()
    output_dir = (args.output_dir or jobs_path.parent / "summary").resolve()
    frame = pd.DataFrame(result_row(job) for job in read_jsonl(jobs_path)).sort_values(
        ["learning_rate", "optimizer"]
    )
    if len(frame) != 10:
        raise ValueError(f"expected ten completed optimizer cells, got {len(frame)}")
    counts = frame.groupby("learning_rate")["optimizer"].nunique()
    if not (counts == 2).all():
        raise ValueError("each learning rate must contain AdamW and Muon")

    contrast_frame = paired_contrasts(frame)
    best_rows = {}
    for optimizer, group in frame.groupby("optimizer"):
        best = group.loc[group["macro_auroc"].idxmax()]
        best_rows[str(optimizer)] = {
            "learning_rate": float(best["learning_rate"]),
            "macro_auroc": float(best["macro_auroc"]),
            "macro_balanced_accuracy": float(best["macro_balanced_accuracy"]),
            "macro_brier": float(best["macro_brier"]),
            "train_loss": float(best["train_loss"]),
            "train_samples_per_second": float(best["train_samples_per_second"]),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "results.csv", index=False)
    contrast_frame.to_csv(output_dir / "paired_contrasts.csv", index=False)
    summary = {
        "selection_split": "internal dataset-label-stratified development set",
        "competition_validation_used_for_selection": False,
        "primary_metric": "macro_auroc",
        "optimizers": ["adamw", "muon"],
        "learning_rates_are_paired": True,
        "muon_adjust_lr_fn": "match_rms_adamw",
        "lr_scheduler_type": "linear",
        "best_by_optimizer": best_rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(frame.to_string(index=False))
    print(contrast_frame.to_string(index=False))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
