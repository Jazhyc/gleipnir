#!/usr/bin/env python3
"""Summarize frozen external-transfer results across adapter seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

METRICS = ("auroc", "balanced_accuracy", "brier")


def result_row(result: dict[str, Any]) -> dict[str, Any]:
    subject = result["metrics"]["subject_macro"]["macro"]
    pooled = result["metrics"]["pooled"]
    row = {
        "target": result["target"],
        "seed": result["seed"],
        "rows": result["rows"],
        "seconds": result["seconds"],
        "subject_groups": len(result["metrics"]["subject_macro"]["groups"]),
    }
    for metric in METRICS:
        row[f"subject_macro_{metric}"] = subject[metric]
        row[f"pooled_{metric}"] = pooled[metric]
    for testbed, view in sorted(result["metrics"]["testbeds"].items()):
        for metric in METRICS:
            row[f"{testbed}_{metric}"] = view["macro"][metric]
    return row


def aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        column
        for column in rows.columns
        if column not in {"target", "seed"} and column != "subject_groups"
    ]
    summary = rows.groupby("target", sort=False)[numeric].mean().reset_index()
    summary.insert(1, "runs", rows.groupby("target", sort=False).size().to_numpy())
    base = summary[summary["target"] == "base"]
    if len(base) != 1:
        raise ValueError("expected exactly one base aggregate")
    base_row = base.iloc[0]
    for metric in (
        "subject_macro_auroc",
        "subject_macro_balanced_accuracy",
        "subject_macro_brier",
    ):
        summary[f"delta_vs_base_{metric}"] = summary[metric] - base_row[metric]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        type=Path,
        default=Path("results/aisi_lie_detection_transfer/runs"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/aisi_lie_detection_transfer/summary"),
    )
    args = parser.parse_args()
    paths = sorted(args.runs.resolve().glob("*/*/result.json"))
    if len(paths) != 7:
        raise ValueError(f"expected seven frozen result files, found {len(paths)}")
    rows = pd.DataFrame([result_row(json.loads(path.read_text())) for path in paths])
    order = {"base": 0, "hard-only": 1, "0500": 2}
    rows = rows.sort_values(
        ["target", "seed"],
        key=lambda values: values.map(order) if values.name == "target" else values,
    ).reset_index(drop=True)
    summary = aggregate(rows)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output / "results.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    payload = {
        "selection": "frozen comparison; no external-result tuning",
        "rows": rows.to_dict(orient="records"),
        "summary": summary.to_dict(orient="records"),
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
