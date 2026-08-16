#!/usr/bin/env python3
"""Run the four-job hard-anchor seed replication on two Lambda H100s."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.adapter_capacity_scaling.prepare import (  # noqa: E402
    read_jsonl,
    write_jsonl,
)
from experiments.adapter_capacity_scaling.run_lambda import (  # noqa: E402
    Status,
    run_lora_lane,
)
from experiments.training_procedure_screen.core import (  # noqa: E402
    balanced_lanes,
    validate_hard_anchor_replication_jobs,
)
from experiments.training_procedure_screen.run_lambda import (  # noqa: E402
    run_evaluation_lane,
)
from gleipnir.qwen35_fast_training import (  # noqa: E402
    DEFAULT_FLA_TARGET,
    ensure_fla_kernels,
)


def relocate_jobs(
    source: Path,
    destination: Path,
    student_rows: Path,
    soft_targets: Path,
    selection_manifest: Path,
    development: Path,
) -> list[dict[str, Any]]:
    """Rewrite host-specific paths without changing the frozen jobs."""
    root = destination.parent
    relocated = []
    for job in read_jsonl(source):
        output = root / "runs" / str(job["job_name"])
        selected = dict(job)
        selected.update(
            student_rows=student_rows.resolve().as_posix(),
            soft_targets=soft_targets.resolve().as_posix(),
            selection_manifest=selection_manifest.resolve().as_posix(),
            validation=development.resolve().as_posix(),
            output_dir=output.resolve().as_posix(),
            causal_adapter_dir=(output / "causal_adapter").resolve().as_posix(),
            model_dir=(output / "model").resolve().as_posix(),
        )
        relocated.append(selected)
    validate_hard_anchor_replication_jobs(relocated)
    write_jsonl(destination, relocated)
    return relocated


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path("results/training_procedure_screen/hard_anchor_replication")
    parser.add_argument("--jobs", type=Path, default=root / "jobs.jsonl")
    parser.add_argument("--lambda-jobs", type=Path, default=root / "lambda_jobs.jsonl")
    parser.add_argument(
        "--student-rows",
        type=Path,
        default=Path("data/data_scaling/student_rows.jsonl"),
    )
    parser.add_argument(
        "--soft-targets",
        type=Path,
        default=Path("data/data_scaling/soft_targets.jsonl"),
    )
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=Path("data/training_procedure_screen/train_full_tuning_pool.jsonl"),
    )
    parser.add_argument(
        "--development",
        type=Path,
        default=Path("data/optimization_regularization_screen/development.jsonl"),
    )
    parser.add_argument("--status", type=Path, default=root / "lambda_status.json")
    parser.add_argument(
        "--screen-results",
        type=Path,
        default=Path("results/training_procedure_screen/summary/results.csv"),
    )
    parser.add_argument("--fla-target", type=Path, default=DEFAULT_FLA_TARGET)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--revision", default=os.environ.get("GLEIPNIR_COMMIT"))
    args = parser.parse_args()
    if args.gpus != 2:
        raise ValueError("the frozen replication requires exactly two H100s")

    jobs_path = args.lambda_jobs.resolve()
    development = args.development.resolve()
    jobs = relocate_jobs(
        args.jobs.resolve(),
        jobs_path,
        args.student_rows.resolve(),
        args.soft_targets.resolve(),
        args.selection_manifest.resolve(),
        development,
    )
    lanes = balanced_lanes(jobs, args.gpus)
    status = Status(
        args.status.resolve(),
        jobs,
        run_metadata={
            "revision": args.revision,
            "phase": "kernel_preflight",
            "cells": len(jobs),
            "training_rows": int(jobs[0]["train_rows"]),
            "development_rows": 1972,
            "anchor_weights": [0.1, 0.25],
            "seeds_added": [1, 2],
            "competition_validation_used": False,
            "final_test_used": False,
        },
    )
    try:
        if args.revision is not None:
            os.environ["GLEIPNIR_COMMIT"] = args.revision
        os.environ.update(ensure_fla_kernels(args.fla_target))
        status.set_phase("hard_anchor_replication_training")
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(run_lora_lane, lane, lane_jobs, jobs_path, status)
                for lane, lane_jobs in enumerate(lanes)
            ]
            for future in futures:
                future.result()

        status.set_phase("causal_checkpoint_evaluation")
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    run_evaluation_lane, lane, lane_jobs, jobs_path, development
                )
                for lane, lane_jobs in enumerate(lanes)
            ]
            for future in futures:
                future.result()

        status.set_phase("summarizing")
        subprocess.run(
            [
                sys.executable,
                "experiments/training_procedure_screen/"
                "summarize_hard_anchor_replication.py",
                "--jobs",
                jobs_path.as_posix(),
                "--screen-results",
                args.screen_results.resolve().as_posix(),
            ],
            cwd=ROOT,
            check=True,
        )
    except BaseException as error:
        status.fail(error)
        raise
    status.finish()


if __name__ == "__main__":
    main()
