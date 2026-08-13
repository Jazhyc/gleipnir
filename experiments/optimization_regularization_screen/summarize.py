#!/usr/bin/env python3
"""Summarize the single-seed optimization and regularization screen."""

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
from experiments.optimization_regularization_screen.core import (  # noqa: E402
    BASELINE_JOB_NAME,
    validate_screen_jobs,
)

METRICS = [
    "macro_auroc",
    "macro_balanced_accuracy",
    "macro_brier",
    "pooled_auroc",
    "pooled_balanced_accuracy",
    "pooled_brier",
]


def result_row(job: dict[str, Any]) -> dict[str, Any]:
    result_path = Path(job["output_dir"]) / "validation" / "result.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"missing internal-development result: {result_path}")
    result = json.loads(result_path.read_text())
    diagnostics = result["metrics"]["macro"]["diagnostics"]
    training = json.loads(
        (Path(job["model_dir"]) / "training_metadata.json").read_text()
    )
    train_metrics = training["train_metrics"]
    training_state = training["training_state"]
    return {
        "job_name": job["job_name"],
        "intervention_family": job["intervention_family"],
        "learning_rate": float(job["learning_rate"]),
        "lr_scheduler_type": job["lr_scheduler_type"],
        "warmup_ratio": float(job["warmup_ratio"]),
        "num_train_epochs": float(job["num_train_epochs"]),
        "lora_dropout": float(job["lora_dropout"]),
        "weight_decay": float(job["weight_decay"]),
        "dataset_sampling": job["dataset_sampling"],
        "soft_target_logit_scale": float(job["soft_target_logit_scale"]),
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
        "global_steps": int(training_state["global_step"]),
        "retained_checkpoints": len(training_state["retained_checkpoints"]),
        "peak_cuda_memory_allocated_bytes": training[
            "peak_cuda_memory_allocated_bytes"
        ],
    }


def contrast_frame(frame: pd.DataFrame) -> pd.DataFrame:
    baseline_rows = frame[frame["job_name"] == BASELINE_JOB_NAME]
    if len(baseline_rows) != 1:
        raise ValueError("screen must contain exactly one baseline result")
    baseline = baseline_rows.iloc[0]
    rows = []
    for result in frame.itertuples():
        row: dict[str, Any] = {
            "job_name": result.job_name,
            "intervention_family": result.intervention_family,
        }
        for metric in METRICS:
            row[f"difference_from_baseline_{metric}"] = float(
                getattr(result, metric) - baseline[metric]
            )
        row["eligible"] = bool(
            row["difference_from_baseline_macro_balanced_accuracy"] >= -0.002
            and row["difference_from_baseline_macro_brier"] <= 0.002
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summary_payload(
    frame: pd.DataFrame, contrasts: pd.DataFrame
) -> dict[str, Any]:
    merged = frame.merge(
        contrasts[["job_name", "eligible"]], on="job_name", validate="one_to_one"
    )
    eligible = merged[merged["eligible"]].sort_values(
        ["macro_auroc", "macro_balanced_accuracy"], ascending=False
    )
    candidates = eligible[eligible["job_name"] != BASELINE_JOB_NAME].head(5)
    baseline = frame[frame["job_name"] == BASELINE_JOB_NAME].iloc[0]
    return {
        "selection_split": "fixed internal dataset-label-stratified development",
        "competition_validation_used": False,
        "final_test_used": False,
        "primary_metric": "macro_auroc",
        "metric_constraints": {
            "macro_balanced_accuracy_max_regression": 0.002,
            "macro_brier_max_regression": 0.002,
        },
        "baseline": {
            "job_name": BASELINE_JOB_NAME,
            "macro_auroc": float(baseline["macro_auroc"]),
            "macro_balanced_accuracy": float(
                baseline["macro_balanced_accuracy"]
            ),
            "macro_brier": float(baseline["macro_brier"]),
        },
        "replication_shortlist": [
            {
                "job_name": row.job_name,
                "intervention_family": row.intervention_family,
                "macro_auroc": float(row.macro_auroc),
                "macro_balanced_accuracy": float(row.macro_balanced_accuracy),
                "macro_brier": float(row.macro_brier),
            }
            for row in candidates.itertuples()
        ],
        "screen_is_single_seed": True,
        "promotion_permitted": False,
        "next_step": (
            "replicate shortlisted cells with seeds 1 and 2 before combining "
            "interventions or training on more data"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jobs",
        type=Path,
        default=Path(
            "results/optimization_regularization_screen/lambda_jobs.jsonl"
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    jobs_path = args.jobs.resolve()
    jobs = read_jsonl(jobs_path)
    validate_screen_jobs(jobs)
    output_dir = (args.output_dir or jobs_path.parent / "summary").resolve()
    frame = pd.DataFrame(result_row(job) for job in jobs).sort_values(
        ["intervention_family", "job_name"]
    )
    contrasts = contrast_frame(frame)
    summary = summary_payload(frame, contrasts)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "results.csv", index=False)
    contrasts.to_csv(output_dir / "contrasts.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(frame.to_string(index=False))
    print(contrasts.to_string(index=False))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
