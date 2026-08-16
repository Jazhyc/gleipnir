#!/usr/bin/env python3
"""Prepare the frozen three-seed hard-label strength curve."""

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
    hard_label_strength_variants,
    validate_hard_label_strength_jobs,
)


def strength_jobs(
    screen_jobs: list[dict[str, object]], output_dir: Path
) -> list[dict[str, object]]:
    """Clone same-seed baselines and change only the declared loss strength."""
    baseline_by_seed = {
        int(job["seed"]): job
        for job in screen_jobs
        if job["intervention_family"] == "baseline"
    }
    if sorted(baseline_by_seed) != [0, 1, 2]:
        raise ValueError("source screen must contain baseline seeds 0, 1, and 2")
    jobs: list[dict[str, object]] = []
    for variant in hard_label_strength_variants():
        job = {**baseline_by_seed[int(variant["seed"])], **variant}
        job_dir = output_dir / "runs" / str(job["job_name"])
        job.update(
            output_dir=job_dir.as_posix(),
            causal_adapter_dir=(job_dir / "causal_adapter").as_posix(),
            model_dir=(job_dir / "model").as_posix(),
        )
        jobs.append(job)
    validate_hard_label_strength_jobs(jobs)
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
        default=Path("results/training_procedure_screen/hard_label_strength"),
    )
    args = parser.parse_args()
    source_path = args.screen_jobs.resolve()
    output_dir = args.output_dir.resolve()
    jobs = strength_jobs(read_jsonl(source_path), output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs_path = output_dir / "jobs.jsonl"
    write_jsonl(jobs_path, jobs)
    manifest = {
        "hypothesis": (
            "hard-label anchoring follows a smooth three-seed response curve with "
            "an interior optimum before hard-label overfitting dominates"
        ),
        "new_jobs": len(jobs),
        "new_strengths": sorted(
            {(job["strength_id"], job["hard_label_fraction"]) for job in jobs}
        ),
        "reused_direct_loss_weights": [0.0, 0.1, 0.25],
        "seeds": [0, 1, 2],
        "source_screen_jobs_sha256": sha256_file(source_path),
        "primary_metric": "three-seed mean paired macro_auroc difference",
        "secondary_metrics": ["macro_balanced_accuracy", "macro_brier"],
        "selection_rule": (
            "select the predeclared strength with maximum three-seed mean macro "
            "AUROC subject to mean paired BA regression <=0.002 and mean paired "
            "Brier regression <=0.002; perform no adaptive grid extension"
        ),
        "checkpoint_selection": False,
        "competition_validation_used": False,
        "final_test_used": False,
        "jobs_path": jobs_path.as_posix(),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"prepared {len(jobs)} hard-label strength jobs; jobs={jobs_path}")


if __name__ == "__main__":
    main()
