#!/usr/bin/env python3
"""Parity-gate and evaluate distilled OOD adapters on the reserved H100."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from experiments.adapter_capacity_scaling.run_lambda import runtime_environment
from experiments.tool_trajectory_monitoring.benchmark_distilled_ood import (
    DEFAULT_CONFIG,
)
from experiments.tool_trajectory_monitoring.qwen_reasoning_core import load_json
from gleipnir.qwen35_fast_training import DEFAULT_FLA_TARGET, ensure_fla_kernels

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT = Path("results/tool_trajectory_distillation_ood")


class Status:
    def __init__(self, path: Path, config: dict[str, Any], revision: str | None):
        self.path = path
        self.value = {
            "state": "evaluating",
            "phase": "initializing",
            "started_at_unix": time.time(),
            "revision": revision,
            "campaign_id": config["campaign_id"],
            "planned_jobs": [
                name
                for group in config["model_groups"].values()
                for name in group["expected_jobs"]
            ],
            "completed_jobs": [],
            "active_job": None,
            "parity": {"9b": "pending", "4b": "pending"},
            "serving_backend": "vllm_continuous_batching",
            "primary_metric": config["scoring"]["primary_metric"],
        }
        self.write()

    def update(self, **values: Any) -> None:
        self.value.update(values)
        self.write()

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.value, indent=2, sort_keys=True) + "\n")
        temporary.replace(self.path)


def run(command: list[str], environment: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--status", type=Path, default=DEFAULT_RESULT / "status.json")
    parser.add_argument("--fla-target", type=Path, default=DEFAULT_FLA_TARGET)
    parser.add_argument("--revision", default=os.environ.get("GLEIPNIR_COMMIT"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    result_dir = args.result_dir.resolve()
    status = Status(args.status.resolve(), config, args.revision)
    environment = runtime_environment("0")
    if args.revision:
        environment["GLEIPNIR_COMMIT"] = args.revision
    try:
        status.update(phase="materializing_student_ood")
        run(
            [
                sys.executable,
                "-m",
                "experiments.tool_trajectory_monitoring.prepare_distillation_ood",
            ]
        )
        status.update(phase="kernel_preflight")
        environment.update(ensure_fla_kernels(args.fla_target))
        for model_size in ("9b", "4b"):
            group = config["model_groups"][model_size]
            parity_job = str(group["parity_job"])
            eager_root = result_dir / "parity" / "eager"
            vllm_root = result_dir / "parity" / "vllm"
            report = result_dir / "parity" / f"{model_size}.json"
            status.update(
                phase=f"{model_size}_eager_parity",
                active_job=f"parity-{model_size}",
            )
            run(
                [
                    sys.executable,
                    "experiments/tool_trajectory_monitoring/evaluate_distilled_ood_causal.py",
                    "--config",
                    config_path.as_posix(),
                    "--model-size",
                    model_size,
                    "--output-root",
                    eager_root.as_posix(),
                ],
                environment,
            )
            status.update(phase=f"{model_size}_vllm_parity")
            run(
                [
                    sys.executable,
                    "experiments/tool_trajectory_monitoring/benchmark_distilled_ood.py",
                    "--config",
                    config_path.as_posix(),
                    "--model-size",
                    model_size,
                    "--output-root",
                    vllm_root.as_posix(),
                    "--only-job",
                    parity_job,
                    "--include-base",
                    "--canary-only",
                ],
                environment,
            )
            status.update(phase=f"{model_size}_parity_comparison")
            run(
                [
                    sys.executable,
                    "experiments/tool_trajectory_monitoring/compare_distilled_ood_parity.py",
                    "--eager-root",
                    eager_root.as_posix(),
                    "--vllm-root",
                    vllm_root.as_posix(),
                    "--model-size",
                    model_size,
                    "--job-name",
                    parity_job,
                    "--output",
                    report.as_posix(),
                ]
            )
            parity = dict(status.value["parity"])
            parity[model_size] = "passed"
            status.update(
                phase=f"{model_size}_full_ood",
                parity=parity,
                active_job=f"full-{model_size}",
            )
            run(
                [
                    sys.executable,
                    "experiments/tool_trajectory_monitoring/benchmark_distilled_ood.py",
                    "--config",
                    config_path.as_posix(),
                    "--model-size",
                    model_size,
                    "--output-root",
                    (result_dir / "runs").as_posix(),
                ],
                environment,
            )
            completed = list(status.value["completed_jobs"])
            completed.extend(group["expected_jobs"])
            status.update(completed_jobs=completed, active_job=None)
        status.update(phase="summarizing")
        run(
            [
                sys.executable,
                "experiments/tool_trajectory_monitoring/summarize_distilled_ood.py",
                "--config",
                config_path.as_posix(),
                "--runs",
                (result_dir / "runs").as_posix(),
                "--output-dir",
                (result_dir / "summary").as_posix(),
            ]
        )
        status.update(
            state="complete",
            phase="complete",
            active_job=None,
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
