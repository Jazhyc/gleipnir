#!/usr/bin/env python3
"""Aggregate adapter-capacity replicates and fit QLoRA-rank scaling curves."""

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
from gleipnir.scaling import fit_bounded_power_law  # noqa: E402


def macro_metric(result: dict[str, Any], name: str) -> float:
    value = result["metrics"]["macro"]["macro"][name]
    if value is None:
        raise ValueError(f"result has no macro {name}")
    return float(value)


def safe_fit(x: pd.Series, y: pd.Series) -> dict[str, Any]:
    try:
        return {"status": "fit", **fit_bounded_power_law(x, y)}
    except ValueError as error:
        return {"status": "rejected", "error": str(error)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jobs",
        type=Path,
        default=Path("results/adapter_capacity_scaling_qlora/lambda_jobs.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--allow-missing-full",
        action="store_true",
        help="Accept intentionally cancelled full-finetuning endpoints.",
    )
    args = parser.parse_args()

    jobs_path = args.jobs.resolve()
    output_dir = (args.output_dir or jobs_path.parent / "summary").resolve()
    rows = []
    for job in read_jsonl(jobs_path):
        result_path = Path(job["output_dir"]) / "validation" / "result.json"
        if not result_path.is_file():
            if args.allow_missing_full and job["capacity_kind"] == "full":
                continue
            raise FileNotFoundError(f"missing evaluation result: {result_path}")
        result = json.loads(result_path.read_text())
        diagnostics = result["metrics"]["macro"]["diagnostics"]
        rows.append(
            {
                "job_name": job["job_name"],
                "capacity_kind": job["capacity_kind"],
                "seed": int(job["seed"]),
                "rank": job["rank"],
                "lora_alpha": job["lora_alpha"],
                "trainable_parameters": int(result["trainable_parameters"]),
                "total_parameters": int(result["total_parameters"]),
                "macro_auroc": macro_metric(result, "auroc"),
                "macro_balanced_accuracy": macro_metric(
                    result, "balanced_accuracy"
                ),
                "macro_brier": macro_metric(result, "brier"),
                "pooled_auroc": float(result["metrics"]["pooled"]["auroc"]),
                "pooled_balanced_accuracy": float(
                    result["metrics"]["pooled"]["balanced_accuracy"]
                ),
                "pooled_brier": float(result["metrics"]["pooled"]["brier"]),
                "unique_scores": int(diagnostics["unique_scores"]),
                "tied_rows": int(diagnostics["tied_rows"]),
            }
        )
    frame = pd.DataFrame(rows).sort_values(
        ["capacity_kind", "rank", "seed"], na_position="last"
    )
    grouped = (
        frame.groupby(["capacity_kind", "rank"], dropna=False, sort=False)
        .agg(
            trainable_parameters=("trainable_parameters", "mean"),
            total_parameters=("total_parameters", "mean"),
            seeds=("seed", "count"),
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
            unique_scores_mean=("unique_scores", "mean"),
            tied_rows_mean=("tied_rows", "mean"),
        )
        .reset_index()
    )
    lora = grouped[grouped["capacity_kind"] == "lora"].sort_values("rank")
    fits = {}
    for metric in (
        "macro_auroc",
        "macro_balanced_accuracy",
        "pooled_auroc",
        "pooled_balanced_accuracy",
    ):
        fits[metric] = {
            "by_rank": safe_fit(lora["rank"], lora[f"{metric}_mean"]),
            "by_trainable_parameters": safe_fit(
                lora["trainable_parameters"], lora[f"{metric}_mean"]
            ),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "replicates.csv", index=False)
    grouped.to_csv(output_dir / "capacity_curve.csv", index=False)
    base_path = jobs_path.parent / "runs" / "base" / "validation" / "result.json"
    base = json.loads(base_path.read_text()) if base_path.is_file() else None
    points = (
        grouped.astype(object)
        .where(pd.notna(grouped), None)
        .to_dict(orient="records")
    )
    summary = {
        "primary_metric": "macro_auroc",
        "balanced_accuracy_threshold": 0.5,
        "fit_population": "QLoRA ranks only; full fine-tuning is an endpoint",
        "fit_form": "metric(r) = asymptote - coefficient * r**(-exponent)",
        "full_finetuning_status": (
            "cancelled_before_start" if args.allow_missing_full else "evaluated"
        ),
        "fits": fits,
        "base_metrics": None if base is None else base["metrics"],
        "points": points,
    }
    (output_dir / "scaling_law.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(grouped.to_string(index=False))
    print(json.dumps(fits, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
