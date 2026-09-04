#!/usr/bin/env python3
"""Summarize the matched inner-mixer compile screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.monitoring_inner_mixer_compile.core import summarize, validate_jobs
from experiments.monitoring_lr_sweep.prepare import atomic_write_json, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    jobs = read_jsonl(args.jobs)
    validate_jobs(jobs)
    for job in jobs:
        metadata_path = Path(job["causal_adapter_dir"]) / "training_metadata.json"
        job["training_metadata"] = json.loads(metadata_path.read_text())
    result = summarize(jobs)
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
