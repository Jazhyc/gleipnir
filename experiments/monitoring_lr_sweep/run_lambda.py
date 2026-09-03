#!/usr/bin/env python3
"""Run the monitoring-only Qwen3.5-4B LR screen on two Lambda H100s."""

from __future__ import annotations

import argparse
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

from experiments.adapter_capacity_scaling.run_lambda import (  # noqa: E402
    runtime_environment,
)
from experiments.monitoring_lr_sweep.core import (  # noqa: E402
    CONTROL_LEARNING_RATE,
    job_name,
    learning_rate_lanes,
    validate_jobs,
    validate_training_metadata,
)
from experiments.tool_trajectory_monitoring.prepare_distillation_scaling import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl,
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
    ensure_qwen35_fast_kernels,
)

DEFAULT_RESULT_DIR = Path("results/monitoring_lr_sweep")


class Status:
    """Thread-safe, resumable campaign status."""

    def __init__(self, path: Path, jobs: list[dict[str, Any]], revision: str | None):
        self.path = path
        self.lock = threading.Lock()
        self.value: dict[str, Any] = {
            "state": "running",
            "phase": "initializing",
            "started_at_unix": time.time(),
            "revision": revision,
            "planned_jobs": [job["job_name"] for job in jobs],
            "completed_jobs": [],
            "active_jobs": [],
            "preflight": "pending",
            "recipe": {
                "data_scope": "monitoring_only",
                "deception_rows": 0,
                "seed": 0,
                "rank": 128,
                "micro_batch_size": 8,
                "gradient_accumulation_steps": 4,
                "effective_batch_size": 32,
                "quantization": "nf4-double-quant-bf16",
                "kernels": "fla-0.5.2+causal-conv1d-1.6.2.post1",
            },
        }
        self.update()

    def update(self, **values: Any) -> None:
        with self.lock:
            self.value.update(values)
            atomic_write_json(self.path, self.value)

    def start(self, name: str) -> None:
        with self.lock:
            if name not in self.value["active_jobs"]:
                self.value["active_jobs"].append(name)
            atomic_write_json(self.path, self.value)

    def finish_job(self, name: str) -> None:
        with self.lock:
            if name in self.value["active_jobs"]:
                self.value["active_jobs"].remove(name)
            if name not in self.value["completed_jobs"]:
                self.value["completed_jobs"].append(name)
            atomic_write_json(self.path, self.value)


def run(command: list[str], environment: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def gpu_training_environment(base: dict[str, str], gpu: int) -> dict[str, str]:
    environment = dict(base)
    cache = f"/tmp/gleipnir-monitoring-lr-gpu-{gpu}"
    environment.update(
        CUDA_VISIBLE_DEVICES=str(gpu),
        TILELANG_CACHE_DIR=f"{cache}/tilelang",
        TORCHINDUCTOR_CACHE_DIR=f"{cache}/torchinductor",
        TRITON_CACHE_DIR=f"{cache}/triton",
        TVM_CACHE_DIR=f"{cache}/tvm",
    )
    return environment


def train_lane(
    lane: list[dict[str, Any]],
    gpu: int,
    jobs_path: Path,
    status: Status,
    environment: dict[str, str],
) -> None:
    for job in lane:
        name = str(job["job_name"])
        status.start(name)
        run_training_job(
            jobs_path,
            name,
            gpu_training_environment(environment, gpu),
            allow_non_scaling_job=True,
        )
        validate_training_metadata(
            Path(job["causal_adapter_dir"]) / "training_metadata.json",
            float(job["learning_rate"]),
        )
        status.finish_job(name)


def parse_args() -> argparse.Namespace:
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
    parser.add_argument(
        "--evaluation-config",
        type=Path,
        default=Path("experiments/monitoring_lr_sweep/id_benchmark.json"),
    )
    parser.add_argument("--fla-target", type=Path, default=DEFAULT_FLA_TARGET)
    parser.add_argument(
        "--causal-conv1d-target", type=Path, default=DEFAULT_CAUSAL_CONV1D_TARGET
    )
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--revision", default=os.environ.get("GLEIPNIR_COMMIT"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gpus != 2:
        raise ValueError("the frozen schedule requires exactly two GPUs")
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
    clean_evaluation_environment = runtime_environment("0")
    try:
        status.update(phase="kernel_preflight")
        fast_environment = ensure_qwen35_fast_kernels(
            args.fla_target, args.causal_conv1d_target
        )
        if args.revision:
            fast_environment["GLEIPNIR_COMMIT"] = args.revision
            clean_evaluation_environment["GLEIPNIR_COMMIT"] = args.revision

        selection = args.preflight_selection.resolve()
        control = next(
            job for job in jobs if float(job["learning_rate"]) == CONTROL_LEARNING_RATE
        )
        preflight_dir = result_dir / "preflight" / "rank128-longest32-mb8"
        preflight_job = {
            **control,
            "job_name": "preflight-r128-longest32-mb8",
            "design_role": "memory_kernel_and_longest_sequence_preflight",
            "train_rows": 32,
            "max_steps": 1,
            "num_train_epochs": -1,
            "selection_manifest": selection.as_posix(),
            "selection_sha256": sha256_file(selection),
            "save_steps": 1,
            "output_dir": preflight_dir.as_posix(),
            "causal_adapter_dir": (preflight_dir / "causal_adapter").as_posix(),
            "model_dir": (preflight_dir / "model").as_posix(),
        }
        preflight_jobs = result_dir / "preflight_jobs.jsonl"
        atomic_write_jsonl(preflight_jobs, [preflight_job])
        verify_job_inputs([preflight_job])
        status.update(phase="longest_sequence_preflight", preflight="running")
        run_training_job(
            preflight_jobs,
            preflight_job["job_name"],
            gpu_training_environment(fast_environment, 0),
            preflight=True,
        )
        validate_training_metadata(
            preflight_dir / "causal_adapter" / "training_metadata.json",
            CONTROL_LEARNING_RATE,
        )
        status.update(phase="training", preflight="passed")
        lanes = learning_rate_lanes(jobs)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    train_lane, lane, gpu, jobs_path, status, fast_environment
                )
                for gpu, lane in enumerate(lanes)
            ]
            for future in futures:
                future.result()

        config = args.evaluation_config.resolve().as_posix()
        eager_root = result_dir / "parity" / "eager"
        vllm_root = result_dir / "parity" / "vllm"
        parity_job = job_name(CONTROL_LEARNING_RATE)
        status.update(phase="eager_parity", active_jobs=[f"parity-{parity_job}"])
        eager_environment = gpu_training_environment(fast_environment, 0)
        run(
            [
                sys.executable,
                "-m",
                "experiments.tool_trajectory_monitoring.evaluate_distilled_ood_causal",
                "--config",
                config,
                "--model-size",
                "4b",
                "--output-root",
                eager_root.as_posix(),
            ],
            eager_environment,
        )
        status.update(phase="vllm_parity")
        run(
            [
                sys.executable,
                "-m",
                "experiments.tool_trajectory_monitoring.benchmark_distilled_ood",
                "--config",
                config,
                "--model-size",
                "4b",
                "--output-root",
                vllm_root.as_posix(),
                "--only-job",
                parity_job,
                "--include-base",
                "--canary-only",
            ],
            clean_evaluation_environment,
        )
        status.update(phase="parity_comparison")
        run(
            [
                sys.executable,
                "-m",
                "experiments.tool_trajectory_monitoring.compare_distilled_ood_parity",
                "--eager-root",
                eager_root.as_posix(),
                "--vllm-root",
                vllm_root.as_posix(),
                "--model-size",
                "4b",
                "--job-name",
                parity_job,
                "--output",
                (result_dir / "parity" / "4b.json").as_posix(),
            ]
        )
        status.update(phase="id_evaluation", active_jobs=["all-five-lr-cells"])
        run(
            [
                sys.executable,
                "-m",
                "experiments.tool_trajectory_monitoring.benchmark_distilled_ood",
                "--config",
                config,
                "--model-size",
                "4b",
                "--output-root",
                (result_dir / "id_evaluation").as_posix(),
            ],
            clean_evaluation_environment,
        )
        status.update(phase="summarizing", active_jobs=[])
        run(
            [
                sys.executable,
                "-m",
                "experiments.monitoring_lr_sweep.summarize",
                "--jobs",
                jobs_path.as_posix(),
                "--runs",
                (result_dir / "id_evaluation").as_posix(),
                "--output-dir",
                (result_dir / "summary").as_posix(),
            ]
        )
        status.update(
            state="complete",
            phase="complete",
            active_jobs=[],
            completed_at_unix=time.time(),
        )
    except BaseException as error:
        status.update(
            state="failed",
            error=repr(error),
            failed_at_unix=time.time(),
        )
        raise


if __name__ == "__main__":
    main()
