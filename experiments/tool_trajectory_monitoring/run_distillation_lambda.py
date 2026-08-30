#!/usr/bin/env python3
"""Preflight and run the seven rank-128 Kimi-soft adapters on one Lambda H100."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from experiments.tool_trajectory_monitoring.distillation_scaling import validate_jobs
from experiments.tool_trajectory_monitoring.prepare_distillation_scaling import (
    atomic_write_jsonl,
    read_jsonl,
)
from gleipnir.qwen35_adapter_rebase import sha256_file
from gleipnir.qwen35_fast_training import DEFAULT_FLA_TARGET, ensure_fla_kernels

ROOT = Path(__file__).resolve().parents[2]


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def relocate_jobs(
    source: Path,
    destination: Path,
    data_dir: Path,
    result_dir: Path,
) -> list[dict[str, Any]]:
    jobs = read_jsonl(source)
    validate_jobs(jobs)
    relocated = []
    for job in jobs:
        selected = dict(job)
        student_name = Path(str(job["student_rows"])).name
        soft_name = Path(str(job["soft_targets"])).name
        selection = job.get("selection_manifest")
        output_dir = result_dir / "runs" / str(job["job_name"])
        selected.update(
            student_rows=(data_dir / student_name).resolve().as_posix(),
            soft_targets=(data_dir / soft_name).resolve().as_posix(),
            selection_manifest=(
                None
                if selection is None
                else (result_dir / "selections" / Path(str(selection)).name)
                .resolve()
                .as_posix()
            ),
            output_dir=output_dir.resolve().as_posix(),
            causal_adapter_dir=(output_dir / "causal_adapter").resolve().as_posix(),
            model_dir=(output_dir / "model").resolve().as_posix(),
        )
        relocated.append(selected)
    atomic_write_jsonl(destination, relocated)
    return relocated


def verify_job_inputs(jobs: list[dict[str, Any]]) -> None:
    checked: dict[Path, str] = {}
    for job in jobs:
        for path_key, checksum_key in (
            ("student_rows", "student_rows_sha256"),
            ("soft_targets", "soft_targets_sha256"),
            ("selection_manifest", "selection_sha256"),
        ):
            value = job.get(path_key)
            if value is None:
                continue
            path = Path(str(value))
            expected = str(job[checksum_key])
            if path in checked and checked[path] == expected:
                continue
            actual = sha256_file(path)
            if actual != expected:
                raise ValueError(f"input checksum drift for {path}: {actual}")
            checked[path] = expected


def runtime_environment() -> dict[str, str]:
    executable_dir = Path(sys.executable).parent.absolute().as_posix()
    current_path = os.environ.get("PATH", "")
    return dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="0",
        PATH=f"{executable_dir}:{current_path}",
        TILELANG_CACHE_DIR="/tmp/gleipnir-tool-distill/tilelang",
        TORCHINDUCTOR_CACHE_DIR="/tmp/gleipnir-tool-distill/torchinductor",
        TRITON_CACHE_DIR="/tmp/gleipnir-tool-distill/triton",
        TVM_CACHE_DIR="/tmp/gleipnir-tool-distill/tvm",
    )


class Status:
    def __init__(self, path: Path, jobs: list[dict[str, Any]], revision: str | None):
        self.path = path
        self.value = {
            "state": "training",
            "phase": "initializing",
            "started_at_unix": time.time(),
            "revision": revision,
            "planned_jobs": [job["job_name"] for job in jobs],
            "completed_jobs": [],
            "active_job": None,
            "preflight": "pending",
            "recipe": {
                "target": "kimi_soft_only",
                "rank": 128,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 32,
                "effective_batch_size": 32,
                "max_length": 29_696,
                "quantization": "nf4-double-quant-bf16",
                "direct_logits_mode": "selected_positions",
            },
        }
        self.write()

    def update(self, **values: Any) -> None:
        self.value.update(values)
        self.write()

    def write(self) -> None:
        atomic_write_json(self.path, self.value)


def run_training_job(
    jobs_path: Path,
    job_name: str,
    environment: dict[str, str],
    *,
    preflight: bool = False,
) -> None:
    command = [
        sys.executable,
        "experiments/tool_trajectory_monitoring/run_distillation_train.py",
        "--jobs",
        jobs_path.as_posix(),
        "--job-name",
        job_name,
    ]
    if preflight:
        command.append("--allow-preflight-job")
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jobs",
        type=Path,
        default=Path("results/tool_trajectory_distillation_scaling/jobs.jsonl"),
    )
    parser.add_argument(
        "--lambda-jobs",
        type=Path,
        default=Path("results/tool_trajectory_distillation_scaling/lambda_jobs.jsonl"),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/tool_trajectory_monitoring/distillation_scaling"),
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("results/tool_trajectory_distillation_scaling"),
    )
    parser.add_argument(
        "--status",
        type=Path,
        default=Path("results/tool_trajectory_distillation_scaling/lambda_status.json"),
    )
    parser.add_argument("--fla-target", type=Path, default=DEFAULT_FLA_TARGET)
    parser.add_argument(
        "--revision", default=os.environ.get("GLEIPNIR_COMMIT")
    )
    args = parser.parse_args()

    result_dir = args.result_dir.resolve()
    jobs_path = args.lambda_jobs.resolve()
    jobs = relocate_jobs(
        args.jobs.resolve(), jobs_path, args.data_dir.resolve(), result_dir
    )
    verify_job_inputs(jobs)
    status = Status(args.status.resolve(), jobs, args.revision)
    environment = runtime_environment()
    try:
        if args.revision:
            environment["GLEIPNIR_COMMIT"] = args.revision
        status.update(phase="kernel_preflight")
        environment.update(ensure_fla_kernels(args.fla_target))

        full_job = next(job for job in jobs if int(job["train_rows"]) == 8_688)
        preflight_selection = (
            result_dir / "selections" / "preflight-longest-32.jsonl"
        ).resolve()
        preflight_dir = (result_dir / "preflight" / "rank128-longest32").resolve()
        preflight_job = {
            **full_job,
            "job_name": "preflight-r128-longest32",
            "design_role": "memory_and_throughput_preflight",
            "train_rows": 32,
            "max_steps": 1,
            "num_train_epochs": -1,
            "selection_manifest": preflight_selection.as_posix(),
            "selection_sha256": sha256_file(preflight_selection),
            "save_steps": 1,
            "output_dir": preflight_dir.as_posix(),
            "causal_adapter_dir": (preflight_dir / "causal_adapter").as_posix(),
            "model_dir": (preflight_dir / "model").as_posix(),
        }
        preflight_jobs_path = result_dir / "preflight_jobs.jsonl"
        atomic_write_jsonl(preflight_jobs_path, [preflight_job])
        status.update(phase="longest_sequence_preflight", preflight="running")
        run_training_job(
            preflight_jobs_path,
            preflight_job["job_name"],
            environment,
            preflight=True,
        )
        status.update(phase="training", preflight="passed")

        completed = []
        for job in jobs:
            job_name = str(job["job_name"])
            status.update(active_job=job_name, completed_jobs=completed)
            run_training_job(jobs_path, job_name, environment)
            completed.append(job_name)
            status.update(active_job=None, completed_jobs=completed)
        status.update(
            state="complete",
            phase="complete",
            active_job=None,
            completed_jobs=completed,
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
