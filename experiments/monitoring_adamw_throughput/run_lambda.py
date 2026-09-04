#!/usr/bin/env python3
"""Preflight fused AdamW, then run both matched conditions on Lambda."""

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

from experiments.monitoring_adamw_throughput.core import (  # noqa: E402
    validate_jobs,
    validate_training_metadata,
)
from experiments.monitoring_lr_sweep.prepare import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl,
)
from experiments.monitoring_lr_sweep.run_lambda import (  # noqa: E402
    gpu_training_environment,
)
from experiments.tool_trajectory_monitoring.run_distillation_lambda import (  # noqa: E402
    relocate_jobs,
    run_training_job,
    verify_job_inputs,
)
from gleipnir.qwen35_adapter_rebase import sha256_file  # noqa: E402
from gleipnir.qwen35_fast_training import (  # noqa: E402
    DEFAULT_CAUSAL_CONV1D_TARGET,
    DEFAULT_FLA_TARGET,
    DEFAULT_TRITON_TARGET,
    ensure_qwen35_long_trajectory_kernels,
)

DEFAULT_RESULT_DIR = Path("results/monitoring_adamw_throughput")


class Status:
    """Thread-safe benchmark status."""

    def __init__(self, path: Path, jobs: list[dict[str, Any]], revision: str | None):
        self.path = path
        self.lock = threading.Lock()
        self.value: dict[str, Any] = {
            "state": "running",
            "phase": "initializing",
            "started_at_unix": time.time(),
            "revision": revision,
            "planned_jobs": [job["job_name"] for job in jobs],
            "active_jobs": [],
            "completed_jobs": [],
            "preflight": "pending",
        }
        self.update()

    def update(self, **values: Any) -> None:
        with self.lock:
            self.value.update(values)
            atomic_write_json(self.path, self.value)

    def start(self, name: str) -> None:
        with self.lock:
            self.value["active_jobs"].append(name)
            atomic_write_json(self.path, self.value)

    def finish(self, name: str) -> None:
        with self.lock:
            self.value["active_jobs"].remove(name)
            self.value["completed_jobs"].append(name)
            atomic_write_json(self.path, self.value)


def run_condition(
    job: dict[str, Any],
    gpu: int,
    jobs_path: Path,
    environment: dict[str, str],
    status: Status,
) -> None:
    name = str(job["job_name"])
    status.start(name)
    run_training_job(
        jobs_path,
        name,
        gpu_training_environment(environment, gpu),
        allow_non_scaling_job=True,
    )
    metadata = json.loads(
        (Path(job["causal_adapter_dir"]) / "training_metadata.json").read_text()
    )
    validate_training_metadata(metadata, job, expected_steps=20)
    status.finish(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, default=DEFAULT_RESULT_DIR / "jobs.jsonl")
    parser.add_argument(
        "--lambda-jobs", type=Path, default=DEFAULT_RESULT_DIR / "lambda_jobs.jsonl"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/tool_trajectory_monitoring/distillation_scaling"),
    )
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument(
        "--status", type=Path, default=DEFAULT_RESULT_DIR / "status.json"
    )
    parser.add_argument(
        "--preflight-selection",
        type=Path,
        default=DEFAULT_RESULT_DIR / "selections" / "preflight-longest-32.jsonl",
    )
    parser.add_argument("--fla-target", type=Path, default=DEFAULT_FLA_TARGET)
    parser.add_argument(
        "--causal-conv1d-target", type=Path, default=DEFAULT_CAUSAL_CONV1D_TARGET
    )
    parser.add_argument("--triton-target", type=Path, default=DEFAULT_TRITON_TARGET)
    parser.add_argument("--revision", default=os.environ.get("GLEIPNIR_COMMIT"))
    args = parser.parse_args()

    result_dir = args.result_dir.resolve()
    jobs_path = args.lambda_jobs.resolve()
    jobs = relocate_jobs(
        args.jobs.resolve(),
        jobs_path,
        args.data_dir.resolve(),
        result_dir,
        validator=validate_jobs,
    )
    verify_job_inputs(jobs)
    status = Status(args.status.resolve(), jobs, args.revision)
    try:
        status.update(phase="kernel_preflight")
        environment = ensure_qwen35_long_trajectory_kernels(
            args.fla_target, args.causal_conv1d_target, args.triton_target
        )
        if args.revision:
            environment["GLEIPNIR_COMMIT"] = args.revision
        fused = next(job for job in jobs if job["job_name"] == "adamw-torch-fused")
        preflight_dir = result_dir / "preflight" / "adamw-torch-fused"
        preflight_selection = args.preflight_selection.resolve()
        preflight = {
            **fused,
            "job_name": "preflight-adamw-torch-fused",
            "design_role": "longest_trajectory_memory_preflight",
            "train_rows": 32,
            "max_steps": 1,
            "selection_manifest": preflight_selection.as_posix(),
            "selection_sha256": sha256_file(preflight_selection),
            "save_steps": 1,
            "output_dir": preflight_dir.as_posix(),
            "causal_adapter_dir": (preflight_dir / "causal_adapter").as_posix(),
            "model_dir": (preflight_dir / "model").as_posix(),
        }
        preflight_jobs = result_dir / "preflight_jobs.jsonl"
        atomic_write_jsonl(preflight_jobs, [preflight])
        verify_job_inputs([preflight])
        status.update(phase="longest_trajectory_preflight", preflight="running")
        run_training_job(
            preflight_jobs,
            str(preflight["job_name"]),
            gpu_training_environment(environment, 0),
            preflight=True,
        )
        preflight_metadata = json.loads(
            (preflight_dir / "causal_adapter" / "training_metadata.json").read_text()
        )
        validate_training_metadata(preflight_metadata, fused, expected_steps=1)

        status.update(phase="benchmarking", preflight="passed")
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    run_condition, job, gpu, jobs_path, environment, status
                )
                for gpu, job in enumerate(jobs)
            ]
            for future in futures:
                future.result()
        status.update(phase="summarizing", active_jobs=[])
        subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.monitoring_adamw_throughput.summarize",
                "--jobs",
                jobs_path.as_posix(),
                "--output",
                (result_dir / "summary.json").as_posix(),
            ],
            cwd=ROOT,
            check=True,
        )
        status.update(
            state="complete", phase="complete", completed_at_unix=time.time()
        )
    except BaseException as error:
        status.update(
            state="failed",
            phase="failed",
            active_jobs=[],
            error=repr(error),
            failed_at_unix=time.time(),
        )
        raise


if __name__ == "__main__":
    main()
