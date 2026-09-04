#!/usr/bin/env python3
"""Preflight and benchmark selective compilation on one Lambda H100."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.monitoring_lr_sweep.prepare import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl,
)
from experiments.monitoring_lr_sweep.run_lambda import (  # noqa: E402
    gpu_training_environment,
)
from experiments.monitoring_selective_compile.core import (  # noqa: E402
    JOB_NAME,
    validate_jobs,
    validate_training_metadata,
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

DEFAULT_RESULT_DIR = Path("results/monitoring_selective_compile")


class Status:
    """Persist the candidate's preflight and benchmark state."""

    def __init__(self, path: Path, revision: str | None):
        self.path = path
        self.value: dict[str, Any] = {
            "state": "running",
            "phase": "initializing",
            "started_at_unix": time.time(),
            "revision": revision,
            "preflight": "pending",
        }
        self.update()

    def update(self, **values: Any) -> None:
        self.value.update(values)
        atomic_write_json(self.path, self.value)


def training_loss(metadata: dict[str, Any]) -> float:
    """Extract a finite Trainer loss for compiled/eager parity."""
    loss = float(metadata.get("train_metrics", {}).get("train_loss", math.nan))
    if not math.isfinite(loss):
        raise ValueError("training metadata has no finite train_loss")
    return loss


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
        "--baseline",
        type=Path,
        default=Path(
            "results/monitoring_selective_checkpointing/runs/"
            "linear-attention-checkpointing/causal_adapter/training_metadata.json"
        ),
    )
    parser.add_argument(
        "--all-layer-baseline",
        type=Path,
        default=Path(
            "results/monitoring_adamw_throughput/runs/adamw-torch/"
            "causal_adapter/training_metadata.json"
        ),
    )
    parser.add_argument(
        "--eager-preflight",
        type=Path,
        default=Path(
            "results/monitoring_selective_checkpointing/preflight/"
            "linear-attention-checkpointing/causal_adapter/training_metadata.json"
        ),
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
    status = Status(args.status.resolve(), args.revision)
    try:
        training_loss(json.loads(args.baseline.resolve().read_text()))
        training_loss(json.loads(args.all_layer_baseline.resolve().read_text()))
        jobs_path = args.lambda_jobs.resolve()
        jobs = relocate_jobs(
            args.jobs.resolve(),
            jobs_path,
            args.data_dir.resolve(),
            result_dir,
            validator=validate_jobs,
        )
        verify_job_inputs(jobs)
        job = jobs[0]
        status.update(phase="kernel_preflight")
        environment = ensure_qwen35_long_trajectory_kernels(
            args.fla_target, args.causal_conv1d_target, args.triton_target
        )
        if args.revision:
            environment["GLEIPNIR_COMMIT"] = args.revision
        environment = gpu_training_environment(environment, 0)
        environment["TORCH_LOGS"] = "recompiles"

        preflight_dir = result_dir / "preflight" / JOB_NAME
        preflight_selection = args.preflight_selection.resolve()
        preflight = {
            **job,
            "job_name": f"preflight-{JOB_NAME}",
            "design_role": "longest_trajectory_compile_preflight",
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
        preflight_environment = dict(environment)
        preflight_environment["TORCHINDUCTOR_CACHE_DIR"] = (
            result_dir / "compile_cache" / "preflight"
        ).as_posix()
        status.update(phase="longest_trajectory_preflight", preflight="running")
        run_training_job(
            preflight_jobs,
            str(preflight["job_name"]),
            preflight_environment,
            preflight=True,
        )
        preflight_metadata = json.loads(
            (preflight_dir / "causal_adapter" / "training_metadata.json").read_text()
        )
        validate_training_metadata(preflight_metadata, expected_steps=1)
        eager_preflight = json.loads(args.eager_preflight.resolve().read_text())
        eager_loss = training_loss(eager_preflight)
        compiled_loss = training_loss(preflight_metadata)
        loss_difference = abs(compiled_loss - eager_loss)
        loss_tolerance = 5e-4 + 1e-3 * abs(eager_loss)
        parity = {
            "eager_loss": eager_loss,
            "compiled_loss": compiled_loss,
            "absolute_difference": loss_difference,
            "tolerance": loss_tolerance,
            "passed": loss_difference <= loss_tolerance,
        }
        atomic_write_json(result_dir / "preflight_parity.json", parity)
        if not parity["passed"]:
            raise ValueError(f"compiled preflight loss parity failed: {parity}")

        benchmark_environment = dict(environment)
        benchmark_environment["TORCHINDUCTOR_CACHE_DIR"] = (
            result_dir / "compile_cache" / "benchmark"
        ).as_posix()
        status.update(phase="benchmarking", preflight="passed")
        run_training_job(
            jobs_path,
            JOB_NAME,
            benchmark_environment,
            allow_non_scaling_job=True,
        )
        candidate = Path(job["causal_adapter_dir"]) / "training_metadata.json"
        validate_training_metadata(json.loads(candidate.read_text()), expected_steps=20)
        status.update(phase="summarizing")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.monitoring_selective_compile.summarize",
                "--baseline",
                args.baseline.resolve().as_posix(),
                "--all-layer-baseline",
                args.all_layer_baseline.resolve().as_posix(),
                "--candidate",
                candidate.as_posix(),
                "--output",
                (result_dir / "summary.json").as_posix(),
            ],
            cwd=ROOT,
            check=True,
        )
        status.update(state="complete", phase="complete", completed_at_unix=time.time())
    except BaseException as error:
        status.update(
            state="failed",
            phase="failed",
            error=repr(error),
            failed_at_unix=time.time(),
        )
        raise


if __name__ == "__main__":
    main()
