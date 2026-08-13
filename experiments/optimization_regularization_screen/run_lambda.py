#!/usr/bin/env python3
"""Run the 16-cell optimization screen on two Lambda H100s."""

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
    runtime_environment,
)
from experiments.optimization_regularization_screen.core import (  # noqa: E402
    balanced_screen_lanes,
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
    """Rewrite machine-local artifact paths while preserving job identity."""
    sweep_root = destination.parent
    relocated = []
    for job in read_jsonl(source):
        job_name = str(job["job_name"])
        output_dir = sweep_root / "runs" / job_name
        selected = dict(job)
        selected.update(
            {
                "student_rows": student_rows.resolve().as_posix(),
                "soft_targets": soft_targets.resolve().as_posix(),
                "selection_manifest": selection_manifest.resolve().as_posix(),
                "validation": development.resolve().as_posix(),
                "output_dir": output_dir.resolve().as_posix(),
                "causal_adapter_dir": (
                    output_dir / "causal_adapter"
                ).resolve().as_posix(),
                "model_dir": (output_dir / "model").resolve().as_posix(),
            }
        )
        relocated.append(selected)
    write_jsonl(destination, relocated)
    return relocated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jobs",
        type=Path,
        default=Path("results/optimization_regularization_screen/jobs.jsonl"),
    )
    parser.add_argument(
        "--lambda-jobs",
        type=Path,
        default=Path(
            "results/optimization_regularization_screen/lambda_jobs.jsonl"
        ),
    )
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
        default=Path(
            "data/optimization_regularization_screen/train_seed0_pool_f050.jsonl"
        ),
    )
    parser.add_argument(
        "--development",
        type=Path,
        default=Path("data/optimization_regularization_screen/development.jsonl"),
    )
    parser.add_argument(
        "--status",
        type=Path,
        default=Path(
            "results/optimization_regularization_screen/lambda_status.json"
        ),
    )
    parser.add_argument("--fla-target", type=Path, default=DEFAULT_FLA_TARGET)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--revision", default=os.environ.get("GLEIPNIR_COMMIT"))
    args = parser.parse_args()
    if args.gpus != 2:
        raise ValueError("the frozen screen schedule requires exactly two H100s")

    jobs_path = args.lambda_jobs.resolve()
    jobs = relocate_jobs(
        args.jobs.resolve(),
        jobs_path,
        args.student_rows.resolve(),
        args.soft_targets.resolve(),
        args.selection_manifest.resolve(),
        args.development.resolve(),
    )
    lanes = balanced_screen_lanes(jobs, args.gpus)
    lane_loads = [
        sum(
            int(job["train_rows"]) * float(job["num_train_epochs"])
            for job in lane
        )
        for lane in lanes
    ]
    status = Status(
        args.status.resolve(),
        jobs,
        run_metadata={
            "revision": args.revision,
            "phase": "kernel_preflight",
            "training_recipe": "qlora-nf4-double-quant-bf16-fla-0.5.2",
            "rank": 128,
            "training_rows": int(jobs[0]["train_rows"]),
            "development": args.development.resolve().as_posix(),
            "development_rows": 1972,
            "cells": len(jobs),
            "lane_epoch_row_loads": lane_loads,
            "optimizer": "adamw",
            "checkpoint_interval": "approximately 0.5 epoch",
            "competition_validation_used": False,
            "final_test_used": False,
        },
    )
    try:
        if args.revision is not None:
            os.environ["GLEIPNIR_COMMIT"] = args.revision
        os.environ.update(ensure_fla_kernels(args.fla_target))
        status.set_phase("optimization_screen_training")
        with ThreadPoolExecutor(max_workers=args.gpus) as executor:
            futures = [
                executor.submit(run_lora_lane, index, lane, jobs_path, status)
                for index, lane in enumerate(lanes)
            ]
            for future in futures:
                future.result()

        status.set_phase("internal_development_evaluation")
        subprocess.run(
            [
                sys.executable,
                "experiments/adapter_capacity_scaling/evaluate.py",
                "--jobs",
                jobs_path.as_posix(),
                "--validation",
                args.development.resolve().as_posix(),
                "--mode",
                "lora",
            ],
            cwd=ROOT,
            env=runtime_environment("0"),
            check=True,
        )
        status.set_phase("summarizing")
        subprocess.run(
            [
                sys.executable,
                "experiments/optimization_regularization_screen/summarize.py",
                "--jobs",
                jobs_path.as_posix(),
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
