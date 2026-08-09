#!/usr/bin/env python3
"""Aggregate seed replicates and fit bounded data-scaling curves."""

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

from experiments.data_scaling.core import fit_bounded_power_law  # noqa: E402
from experiments.data_scaling.prepare import read_jsonl  # noqa: E402


def metric_value(result: dict[str, Any], name: str) -> float:
    value = result["metrics"]["macro"]["macro"][name]
    if value is None:
        raise ValueError(f"result has no macro {name}")
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jobs", type=Path, default=Path("results/data_scaling/jobs.jsonl")
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    jobs_path = args.jobs.resolve()
    output_dir = (args.output_dir or jobs_path.parent / "summary").resolve()
    rows = []
    for job in read_jsonl(jobs_path):
        result_path = Path(job["output_dir"]) / "validation" / "result.json"
        if not result_path.is_file():
            raise FileNotFoundError(f"missing evaluation result: {result_path}")
        result = json.loads(result_path.read_text())
        rows.append(
            {
                "job_name": job["job_name"],
                "seed": int(job["seed"]),
                "fraction": float(job["fraction"]),
                "train_rows": int(job["train_rows"]),
                "macro_auroc": metric_value(result, "auroc"),
                "macro_balanced_accuracy": metric_value(
                    result, "balanced_accuracy"
                ),
                "pooled_auroc": float(result["metrics"]["pooled"]["auroc"]),
                "pooled_balanced_accuracy": float(
                    result["metrics"]["pooled"]["balanced_accuracy"]
                ),
            }
        )
    frame = pd.DataFrame(rows).sort_values(["fraction", "seed"])
    grouped = (
        frame.groupby("fraction", sort=True)
        .agg(
            train_rows=("train_rows", "mean"),
            seeds=("seed", "count"),
            macro_auroc_mean=("macro_auroc", "mean"),
            macro_auroc_std=("macro_auroc", "std"),
            macro_balanced_accuracy_mean=("macro_balanced_accuracy", "mean"),
            macro_balanced_accuracy_std=("macro_balanced_accuracy", "std"),
            pooled_auroc_mean=("pooled_auroc", "mean"),
            pooled_auroc_std=("pooled_auroc", "std"),
            pooled_balanced_accuracy_mean=("pooled_balanced_accuracy", "mean"),
            pooled_balanced_accuracy_std=("pooled_balanced_accuracy", "std"),
        )
        .reset_index()
    )
    fits = {
        metric: fit_bounded_power_law(
            grouped["train_rows"], grouped[f"{metric}_mean"]
        )
        for metric in (
            "macro_auroc",
            "macro_balanced_accuracy",
            "pooled_auroc",
            "pooled_balanced_accuracy",
        )
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "replicates.csv", index=False)
    grouped.to_csv(output_dir / "scaling_curve.csv", index=False)
    summary = {
        "primary_metric": "macro_auroc",
        "balanced_accuracy_threshold": 0.5,
        "fit_form": "metric(N) = asymptote - coefficient * N**(-exponent)",
        "fits": fits,
        "points": grouped.to_dict(orient="records"),
    }
    (output_dir / "scaling_law.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(grouped.to_string(index=False))
    print(json.dumps(fits, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
