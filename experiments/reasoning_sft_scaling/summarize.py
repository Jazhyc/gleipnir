#!/usr/bin/env python3
"""Summarize the corrected single-seed reasoning-SFT scaling curve."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.data_scaling.prepare import read_jsonl  # noqa: E402
from gleipnir.scaling import fit_bounded_power_law  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jobs", type=Path, default=Path("results/reasoning_sft_scaling/jobs.jsonl")
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    jobs_path = args.jobs.resolve()
    output_dir = (args.output_dir or jobs_path.parent / "summary").resolve()
    rows = []
    for job in read_jsonl(jobs_path):
        result_path = (
            jobs_path.parent
            / "runs"
            / job["job_name"]
            / "validation"
            / "result.json"
        )
        result = json.loads(result_path.read_text())
        macro = result["metrics"]["macro"]["macro"]
        rows.append(
            {
                "job_name": job["job_name"],
                "fraction": float(job["fraction"]),
                "train_rows": int(job["train_rows"]),
                "seed": int(job["seed"]),
                "macro_auroc": float(macro["auroc"]),
                "macro_balanced_accuracy": float(macro["balanced_accuracy"]),
                "macro_brier": float(macro["brier"]),
                "pooled_auroc": float(result["metrics"]["pooled"]["auroc"]),
                "parse_errors": int(result["parse_errors"]),
                "seconds": float(result["total_seconds"]),
            }
        )
    frame = pd.DataFrame(rows).sort_values("train_rows")
    fits = {}
    for name in ("macro_auroc", "macro_balanced_accuracy"):
        try:
            fits[name] = {
                "status": "fit",
                **fit_bounded_power_law(frame["train_rows"], frame[name]),
            }
        except ValueError as error:
            fits[name] = {"status": "rejected", "error": str(error)}
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "scaling_curve.csv", index=False)
    (output_dir / "scaling_law.json").write_text(
        json.dumps(
            {
                "primary_metric": "macro_auroc",
                "protocol": "reason-then-terminal-logit",
                "seeds": 1,
                "fits": fits,
                "points": frame.to_dict(orient="records"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
