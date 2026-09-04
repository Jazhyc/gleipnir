#!/usr/bin/env python3
"""Summarize the matched monitoring-training throughput conditions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.monitoring_lr_sweep.prepare import atomic_write_json, read_jsonl
from experiments.monitoring_training_throughput.core import (
    select_recipe,
    summarize_condition,
    validate_jobs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jobs",
        type=Path,
        default=Path("results/monitoring_training_throughput/lambda_jobs.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/monitoring_training_throughput/summary.json"),
    )
    args = parser.parse_args()
    jobs = read_jsonl(args.jobs)
    validate_jobs(jobs)
    rows = []
    for job in jobs:
        metadata_path = Path(job["causal_adapter_dir"]) / "training_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        rows.append(summarize_condition(metadata, job))
    selection = select_recipe(rows)
    result = {"conditions": rows, "selection": selection}
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
