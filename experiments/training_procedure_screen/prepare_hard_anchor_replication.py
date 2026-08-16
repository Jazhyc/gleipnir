#!/usr/bin/env python3
"""Prepare seeds 1 and 2 for the two winning hard-anchor cells."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.adapter_capacity_scaling.prepare import (  # noqa: E402
    read_jsonl,
    sha256_file,
    write_jsonl,
)
from experiments.training_procedure_screen.core import (  # noqa: E402
    hard_anchor_replication_variants,
    validate_hard_anchor_replication_jobs,
)


def replication_jobs(
    screen_jobs: list[dict[str, object]], output_dir: Path
) -> list[dict[str, object]]:
    """Clone the selected seed-0 jobs while changing only identity and seed."""
    source_by_name = {str(job["job_name"]): job for job in screen_jobs}
    jobs: list[dict[str, object]] = []
    for variant in hard_anchor_replication_variants():
        source_name = str(variant["source_job_name"])
        if source_name not in source_by_name:
            raise ValueError(f"missing source screen job: {source_name}")
        job = {**source_by_name[source_name], **variant}
        job_dir = output_dir / "runs" / str(job["job_name"])
        job.update(
            output_dir=job_dir.as_posix(),
            causal_adapter_dir=(job_dir / "causal_adapter").as_posix(),
            model_dir=(job_dir / "model").as_posix(),
        )
        jobs.append(job)
    validate_hard_anchor_replication_jobs(jobs)
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--screen-jobs",
        type=Path,
        default=Path("results/training_procedure_screen/jobs.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/training_procedure_screen/hard_anchor_replication"
        ),
    )
    args = parser.parse_args()
    source_path = args.screen_jobs.resolve()
    output_dir = args.output_dir.resolve()
    jobs = replication_jobs(read_jsonl(source_path), output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs_path = output_dir / "jobs.jsonl"
    write_jsonl(jobs_path, jobs)
    manifest = {
        "hypothesis": (
            "the seed-0 hard-anchor improvement persists across seeds 1 and 2"
        ),
        "intervention": "hard-label loss weights 0.10 and 0.25",
        "job_names": [job["job_name"] for job in jobs],
        "seeds_added": [1, 2],
        "source_screen_jobs_sha256": sha256_file(source_path),
        "primary_metric": "mean paired macro_auroc difference from same-seed baseline",
        "secondary_metrics": ["macro_balanced_accuracy", "macro_brier"],
        "selection_rule": (
            "prefer the anchor with the larger three-seed mean macro AUROC if its "
            "mean BA regression is at most 0.002 and mean Brier regression is at "
            "most 0.002; do not tune a new anchor weight"
        ),
        "competition_validation_used": False,
        "final_test_used": False,
        "jobs_path": jobs_path.as_posix(),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"prepared {len(jobs)} hard-anchor replication jobs; jobs={jobs_path}")


if __name__ == "__main__":
    main()
