#!/usr/bin/env python3
"""Summarize the frozen distilled-monitor OOD scaling campaign."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from experiments.tool_trajectory_monitoring.benchmark_distilled_ood import (
    DEFAULT_CONFIG,
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def summary_rows(config: dict[str, Any], runs: Path) -> list[dict[str, Any]]:
    rows = []
    for model_size, group in config["model_groups"].items():
        prices = group["hosted_price_usd_per_million"]
        for job_name in group["expected_jobs"]:
            path = runs / model_size / "adapters" / job_name / "result.json"
            if not path.is_file():
                raise FileNotFoundError(f"missing completed OOD result: {path}")
            result = read_json(path)
            if int(result.get("rows", -1)) != int(config["scope"]["rows"]):
                raise ValueError(f"incomplete OOD result: {path}")
            macro = result["metrics"]["macro"]["macro"]
            pooled = result["metrics"]["pooled"]
            mean_tokens = float(result["prompt_tokens"]["mean"])
            cost = (
                mean_tokens * float(prices["input"]) + float(prices["output"])
            ) / 1_000
            training_job = result["training_job"]
            rows.append(
                {
                    "model_size": model_size,
                    "model": group["id"],
                    "job_name": job_name,
                    "train_rows": int(training_job["train_rows"]),
                    "mixed_data": "mixed" in job_name,
                    "seed": int(training_job["seed"]),
                    "mean_prompt_tokens": mean_tokens,
                    "uncached_hosted_cost_usd_per_1000": cost,
                    "mean_ood_pauroc_at_20": float(macro["pauroc_at_20"]),
                    "mean_ood_auroc": float(macro["auroc"]),
                    "pooled_pauroc_at_20": float(pooled["pauroc_at_20"]),
                    "pooled_auroc": float(pooled["auroc"]),
                    "result_path": path.as_posix(),
                    "training_metadata_sha256": result["training_metadata_sha256"],
                    "adapter_rebase_source_sha256": result[
                        "adapter_rebase_source_sha256"
                    ],
                    "adapter_rebase_destination_sha256": result[
                        "adapter_rebase_destination_sha256"
                    ],
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--runs",
        type=Path,
        default=Path("results/tool_trajectory_distillation_ood/runs"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/tool_trajectory_distillation_ood/summary"),
    )
    args = parser.parse_args()
    config = read_json(args.config)
    rows = summary_rows(config, args.runs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "campaign_id": config["campaign_id"],
                "primary_metric": config["scoring"]["primary_metric"],
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    with (args.output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['job_name']}: train_rows={row['train_rows']} "
            f"mean_ood_pauroc_at_20={row['mean_ood_pauroc_at_20']:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
