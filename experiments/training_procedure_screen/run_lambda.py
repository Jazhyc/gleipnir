#!/usr/bin/env python3
"""Run the structural training-procedure screen on two Lambda H100s."""

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
from experiments.training_procedure_screen.core import (  # noqa: E402
    balanced_lanes,
    validate_screen_jobs,
)
from gleipnir.qwen35_fast_training import (  # noqa: E402
    DEFAULT_FLA_TARGET,
    ensure_fla_kernels,
)

RISK_PREFLIGHT_JOBS = (
    "base-bf16-lora",
    "init-loftq",
    "adapter-dora",
    "init-eva",
    "targets-with-lm-head",
    "decision-token-rows",
    "decision-binary-head",
    "group-dro",
)


def relocate_jobs(
    source: Path,
    destination: Path,
    student_rows: Path,
    soft_targets: Path,
    selection_manifest: Path,
    development: Path,
) -> list[dict[str, Any]]:
    """Rewrite local artifact paths while preserving frozen identities."""
    root = destination.parent
    relocated = []
    for job in read_jsonl(source):
        output = root / "runs" / str(job["job_name"])
        selected = dict(job)
        selected.update(
            {
                "student_rows": student_rows.resolve().as_posix(),
                "soft_targets": soft_targets.resolve().as_posix(),
                "selection_manifest": selection_manifest.resolve().as_posix(),
                "validation": development.resolve().as_posix(),
                "output_dir": output.resolve().as_posix(),
                "causal_adapter_dir": (output / "causal_adapter").resolve().as_posix(),
                "model_dir": (output / "model").resolve().as_posix(),
            }
        )
        relocated.append(selected)
    validate_screen_jobs(relocated)
    write_jsonl(destination, relocated)
    return relocated


def preflight_jobs(jobs: list[dict[str, Any]], path: Path) -> list[dict[str, Any]]:
    """Create isolated two-step copies of every risky implementation cell."""
    root = path.parent / "preflight" / "runs"
    selected = []
    for job in jobs:
        if job["job_name"] not in RISK_PREFLIGHT_JOBS:
            continue
        copy = dict(job)
        output = root / str(job["job_name"])
        copy.update(
            job_name=f"preflight-{job['job_name']}",
            max_steps=2,
            num_train_epochs=2.0,
            save_strategy="no",
            output_dir=output.as_posix(),
            causal_adapter_dir=(output / "causal_adapter").as_posix(),
            model_dir=(output / "model").as_posix(),
        )
        selected.append(copy)
    write_jsonl(path, selected)
    return selected


def run_preflight_lane(
    lane: int,
    jobs: list[dict[str, Any]],
    jobs_path: Path,
    development: Path,
) -> None:
    for job in jobs:
        environment = runtime_environment(str(lane))
        subprocess.run(
            [
                sys.executable,
                "experiments/adapter_capacity_scaling/run_train.py",
                "--jobs",
                jobs_path.as_posix(),
                "--job-name",
                str(job["job_name"]),
            ],
            cwd=ROOT,
            env=environment,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "experiments/training_procedure_screen/evaluate_causal.py",
                "--jobs",
                jobs_path.as_posix(),
                "--job-name",
                str(job["job_name"]),
                "--validation",
                development.as_posix(),
                "--smoke-rows",
                "8",
            ],
            cwd=ROOT,
            env=environment,
            check=True,
        )


def run_evaluation_lane(
    lane: int, jobs: list[dict[str, Any]], jobs_path: Path, development: Path
) -> None:
    for job in jobs:
        subprocess.run(
            [
                sys.executable,
                "experiments/training_procedure_screen/evaluate_causal.py",
                "--jobs",
                jobs_path.as_posix(),
                "--job-name",
                str(job["job_name"]),
                "--validation",
                development.as_posix(),
            ],
            cwd=ROOT,
            env=runtime_environment(str(lane)),
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jobs",
        type=Path,
        default=Path("results/training_procedure_screen/jobs.jsonl"),
    )
    parser.add_argument(
        "--lambda-jobs",
        type=Path,
        default=Path("results/training_procedure_screen/lambda_jobs.jsonl"),
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
        default=Path("data/training_procedure_screen/train_full_tuning_pool.jsonl"),
    )
    parser.add_argument(
        "--development",
        type=Path,
        default=Path("data/optimization_regularization_screen/development.jsonl"),
    )
    parser.add_argument(
        "--status",
        type=Path,
        default=Path("results/training_procedure_screen/lambda_status.json"),
    )
    parser.add_argument("--fla-target", type=Path, default=DEFAULT_FLA_TARGET)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--revision", default=os.environ.get("GLEIPNIR_COMMIT"))
    args = parser.parse_args()
    if args.gpus != 2:
        raise ValueError("the frozen campaign requires exactly two H100s")

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
            "baseline_seeds": [0, 1, 2],
            "rank": 128,
            "risk_preflights": list(RISK_PREFLIGHT_JOBS),
            "competition_validation_used": False,
            "final_test_used": False,
        },
    )
    try:
        if args.revision is not None:
            os.environ["GLEIPNIR_COMMIT"] = args.revision
        os.environ.update(ensure_fla_kernels(args.fla_target))
        if not args.skip_preflight:
            status.set_phase("risk_cell_preflights")
            preflight_path = jobs_path.parent / "preflight_jobs.jsonl"
            risk_jobs = preflight_jobs(jobs, preflight_path)
            risk_lanes = [risk_jobs[::2], risk_jobs[1::2]]
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(
                        run_preflight_lane,
                        lane,
                        lane_jobs,
                        preflight_path,
                        development,
                    )
                    for lane, lane_jobs in enumerate(risk_lanes)
                ]
                for future in futures:
                    future.result()
        if args.preflight_only:
            status.finish()
            return

        status.set_phase("training_procedure_screen_training")
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
                "experiments/training_procedure_screen/summarize.py",
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
