#!/usr/bin/env python3
"""Run the prepared scaling sweep across a fixed set of Lambda GPUs."""

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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def relocate_jobs(
    source: Path,
    destination: Path,
    student_rows: Path,
    soft_targets: Path,
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
                "selection_manifest": (
                    sweep_root
                    / "selections"
                    / Path(job["selection_manifest"]).name
                ).resolve().as_posix(),
                "output_dir": output_dir.resolve().as_posix(),
                "adapter_dir": (output_dir / "adapter").resolve().as_posix(),
            }
        )
        relocated.append(selected)
    write_jsonl(destination, relocated)
    return relocated


def balanced_lanes(
    jobs: list[dict[str, Any]], lane_count: int
) -> list[list[tuple[int, dict[str, Any]]]]:
    """Use longest-processing-time assignment with training rows as the proxy."""
    if lane_count < 1:
        raise ValueError("lane_count must be positive")
    lanes: list[list[tuple[int, dict[str, Any]]]] = [[] for _ in range(lane_count)]
    loads = [0] * lane_count
    indexed = sorted(
        enumerate(jobs),
        key=lambda item: (-int(item[1]["train_rows"]), item[0]),
    )
    for item in indexed:
        lane = min(range(lane_count), key=lambda index: (loads[index], index))
        lanes[lane].append(item)
        loads[lane] += int(item[1]["train_rows"])
    return lanes


class Status:
    def __init__(
        self,
        path: Path,
        lanes: list[list[tuple[int, dict[str, Any]]]],
    ) -> None:
        self.path = path
        self.lock = threading.Lock()
        self.value: dict[str, Any] = {
            "state": "training",
            "started_at_unix": time.time(),
            "lanes": {
                str(index): {
                    "planned_jobs": [job["job_name"] for _, job in lane],
                    "planned_rows": sum(int(job["train_rows"]) for _, job in lane),
                    "completed_jobs": [],
                    "active_job": None,
                }
                for index, lane in enumerate(lanes)
            },
        }
        self._write()

    def update_lane(self, lane: int, **values: Any) -> None:
        with self.lock:
            self.value["lanes"][str(lane)].update(values)
            self._write()

    def finish_job(self, lane: int, job_name: str) -> None:
        with self.lock:
            lane_status = self.value["lanes"][str(lane)]
            lane_status["completed_jobs"].append(job_name)
            lane_status["active_job"] = None
            self._write()

    def set_state(self, state: str, **values: Any) -> None:
        with self.lock:
            self.value.update({"state": state, **values})
            self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.value, indent=2, sort_keys=True) + "\n")
        temporary.replace(self.path)


def run_lane(
    lane_index: int,
    lane: list[tuple[int, dict[str, Any]]],
    jobs_path: Path,
    status: Status,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
) -> None:
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(lane_index))
    for job_index, job in lane:
        status.update_lane(lane_index, active_job=job["job_name"])
        subprocess.run(
            [
                sys.executable,
                "experiments/data_scaling/run_train.py",
                "--jobs",
                jobs_path.as_posix(),
                "--index",
                str(job_index),
                "--per-device-train-batch-size",
                str(micro_batch_size),
                "--gradient-accumulation-steps",
                str(gradient_accumulation_steps),
            ],
            cwd=ROOT,
            env=environment,
            check=True,
        )
        status.finish_job(lane_index, str(job["job_name"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jobs", type=Path, default=Path("results/data_scaling/jobs.jsonl")
    )
    parser.add_argument(
        "--lambda-jobs",
        type=Path,
        default=Path("results/data_scaling/lambda_jobs.jsonl"),
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
        default=Path("results/data_scaling/lambda_status.json"),
    )
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    args = parser.parse_args()

    jobs_path = args.lambda_jobs.resolve()
    jobs = relocate_jobs(
        args.jobs.resolve(),
        jobs_path,
        args.student_rows.resolve(),
        args.soft_targets.resolve(),
    )
    lanes = balanced_lanes(jobs, args.gpus)
    status = Status(args.status.resolve(), lanes)
    try:
        with ThreadPoolExecutor(max_workers=args.gpus) as executor:
            futures = [
                executor.submit(
                    run_lane,
                    index,
                    lane,
                    jobs_path,
                    status,
                    args.per_device_train_batch_size,
                    args.gradient_accumulation_steps,
                )
                for index, lane in enumerate(lanes)
            ]
            for future in futures:
                future.result()
        status.set_state("evaluating", training_finished_at_unix=time.time())
        evaluation_environment = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
        subprocess.run(
            [
                sys.executable,
                "experiments/data_scaling/evaluate.py",
                "--jobs",
                jobs_path.as_posix(),
                "--validation",
                args.validation.resolve().as_posix(),
            ],
            cwd=ROOT,
            env=evaluation_environment,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "experiments/data_scaling/summarize.py",
                "--jobs",
                jobs_path.as_posix(),
            ],
            cwd=ROOT,
            check=True,
        )
    except BaseException as error:
        status.set_state("failed", error=repr(error), failed_at_unix=time.time())
        raise
    status.set_state("complete", completed_at_unix=time.time())


if __name__ == "__main__":
    main()
