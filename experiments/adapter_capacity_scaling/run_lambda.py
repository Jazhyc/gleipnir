#!/usr/bin/env python3
"""Run the capacity sweep on the reserved two-H100 Lambda instance."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.adapter_capacity_scaling.core import (  # noqa: E402
    balanced_lora_lanes,
    split_jobs,
)
from experiments.adapter_capacity_scaling.prepare import (  # noqa: E402
    read_jsonl,
    write_jsonl,
)


def relocate_jobs(
    source: Path,
    destination: Path,
    student_rows: Path,
    soft_targets: Path,
    validation: Path,
) -> list[dict[str, Any]]:
    """Rewrite machine-local artifact paths while preserving job identities."""
    jobs = read_jsonl(source)
    sweep_root = destination.parent
    relocated = []
    for job in jobs:
        job_name = str(job["job_name"])
        output_dir = sweep_root / "runs" / job_name
        selected = dict(job)
        selected.update(
            {
                "student_rows": student_rows.resolve().as_posix(),
                "soft_targets": soft_targets.resolve().as_posix(),
                "validation": validation.resolve().as_posix(),
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


class Status:
    def __init__(
        self,
        path: Path,
        jobs: list[dict[str, Any]],
        cancelled_jobs: list[dict[str, Any]] | None = None,
    ) -> None:
        self.path = path
        self.lock = threading.Lock()
        self.value: dict[str, Any] = {
            "state": "training",
            "phase": "lora_training",
            "started_at_unix": time.time(),
            "planned_jobs": [job["job_name"] for job in jobs],
            "cancelled_jobs": [
                job["job_name"] for job in (cancelled_jobs or [])
            ],
            "completed_jobs": [],
            "active_jobs": [],
        }
        self._write()

    def set_phase(self, phase: str) -> None:
        with self.lock:
            self.value["phase"] = phase
            self._write()

    def start_job(self, job_name: str) -> None:
        with self.lock:
            active = self.value["active_jobs"]
            if job_name not in active:
                active.append(job_name)
            self._write()

    def finish_job(self, job_name: str) -> None:
        with self.lock:
            active = self.value["active_jobs"]
            if job_name in active:
                active.remove(job_name)
            completed = self.value["completed_jobs"]
            if job_name not in completed:
                completed.append(job_name)
            self._write()

    def finish(self) -> None:
        with self.lock:
            self.value.update(
                state="complete",
                phase="complete",
                completed_at_unix=time.time(),
                active_jobs=[],
            )
            self._write()

    def fail(self, error: BaseException) -> None:
        with self.lock:
            self.value.update(
                state="failed",
                error=repr(error),
                failed_at_unix=time.time(),
            )
            self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.value, indent=2, sort_keys=True) + "\n")
        temporary.replace(self.path)


def runtime_environment(cuda_visible_devices: str) -> dict[str, str]:
    executable_dir = Path(sys.executable).resolve().parent.as_posix()
    current_path = os.environ.get("PATH", "")
    return dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=cuda_visible_devices,
        PATH=f"{executable_dir}:{current_path}",
    )


def run_job(
    job: dict[str, Any],
    jobs_path: Path,
    status: Status,
    cuda_visible_devices: str,
    distributed_processes: int = 2,
) -> None:
    job_name = str(job["job_name"])
    status.start_job(job_name)
    subprocess.run(
        [
            sys.executable,
            "experiments/adapter_capacity_scaling/run_train.py",
            "--jobs",
            jobs_path.as_posix(),
            "--job-name",
            job_name,
            "--distributed-processes",
            str(distributed_processes),
        ],
        cwd=ROOT,
        env=runtime_environment(cuda_visible_devices),
        check=True,
    )
    status.finish_job(job_name)


def run_lora_lane(
    lane_index: int,
    jobs: list[dict[str, Any]],
    jobs_path: Path,
    status: Status,
) -> None:
    for job in jobs:
        run_job(job, jobs_path, status, str(lane_index))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jobs",
        type=Path,
        default=Path("results/adapter_capacity_scaling_causal/jobs.jsonl"),
    )
    parser.add_argument(
        "--lambda-jobs",
        type=Path,
        default=Path("results/adapter_capacity_scaling_causal/lambda_jobs.jsonl"),
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
        "--validation",
        type=Path,
        default=Path("data/data_scaling/validation.jsonl"),
    )
    parser.add_argument(
        "--status",
        type=Path,
        default=Path("results/adapter_capacity_scaling_causal/lambda_status.json"),
    )
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument(
        "--skip-full",
        action="store_true",
        help="Run and evaluate LoRAs only; record full fine-tunes as cancelled.",
    )
    args = parser.parse_args()
    if args.gpus != 2:
        raise ValueError("this schedule is predeclared for exactly two H100 GPUs")

    jobs_path = args.lambda_jobs.resolve()
    jobs = relocate_jobs(
        args.jobs.resolve(),
        jobs_path,
        args.student_rows.resolve(),
        args.soft_targets.resolve(),
        args.validation.resolve(),
    )
    lora_jobs, full_jobs = split_jobs(jobs)
    full_jobs.sort(key=lambda job: int(job["seed"]))
    if len(full_jobs) != 3:
        raise ValueError(f"expected three full fine-tuning jobs, got {len(full_jobs)}")
    scheduled_jobs = lora_jobs if args.skip_full else jobs
    cancelled_jobs = full_jobs if args.skip_full else []
    status = Status(args.status.resolve(), scheduled_jobs, cancelled_jobs)

    try:
        lanes = balanced_lora_lanes(lora_jobs, args.gpus)
        with ThreadPoolExecutor(max_workers=args.gpus) as executor:
            futures = [
                executor.submit(run_lora_lane, index, lane, jobs_path, status)
                for index, lane in enumerate(lanes)
            ]
            for future in futures:
                future.result()

        if not args.skip_full:
            status.set_phase("full_training")
            for job in full_jobs:
                run_job(job, jobs_path, status, "0,1", args.gpus)

        status.set_phase("lora_evaluation")
        subprocess.run(
            [
                sys.executable,
                "experiments/adapter_capacity_scaling/evaluate.py",
                "--jobs",
                jobs_path.as_posix(),
                "--validation",
                args.validation.resolve().as_posix(),
                "--mode",
                "lora",
            ],
            cwd=ROOT,
            env=runtime_environment("0"),
            check=True,
        )

        if not args.skip_full:
            status.set_phase("full_evaluation")
            for job in full_jobs:
                subprocess.run(
                    [
                        sys.executable,
                        "experiments/adapter_capacity_scaling/evaluate.py",
                        "--jobs",
                        jobs_path.as_posix(),
                        "--validation",
                        args.validation.resolve().as_posix(),
                        "--mode",
                        "full",
                        "--job-name",
                        str(job["job_name"]),
                    ],
                    cwd=ROOT,
                    env=runtime_environment("0"),
                    check=True,
                )

        status.set_phase("summarizing")
        summarize_command = [
            sys.executable,
            "experiments/adapter_capacity_scaling/summarize.py",
            "--jobs",
            jobs_path.as_posix(),
        ]
        if args.skip_full:
            summarize_command.append("--allow-missing-full")
        subprocess.run(
            summarize_command,
            cwd=ROOT,
            check=True,
        )
    except BaseException as error:
        status.fail(error)
        raise
    status.finish()


if __name__ == "__main__":
    main()
