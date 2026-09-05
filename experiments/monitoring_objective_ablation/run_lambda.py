#!/usr/bin/env python3
"""Run and evaluate the monitoring objective screen on two Lambda H100s."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
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
from experiments.monitoring_objective_ablation.core import (  # noqa: E402
    CAMPAIGN_ID,
    validate_jobs,
    validate_training_metadata,
)
from experiments.tool_trajectory_monitoring.benchmark_distilled_ood import (  # noqa: E402
    validate_config as validate_evaluation_config,
)
from experiments.tool_trajectory_monitoring.benchmark_distilled_ood import (  # noqa: E402
    validate_inputs as validate_evaluation_inputs,
)
from experiments.tool_trajectory_monitoring.benchmark_distilled_ood import (  # noqa: E402
    validate_jobs as validate_evaluation_jobs,
)
from experiments.tool_trajectory_monitoring.prepare_distillation_scaling import (  # noqa: E402
    atomic_write_jsonl,
    read_jsonl,
)
from experiments.tool_trajectory_monitoring.run_distillation_lambda import (  # noqa: E402
    run_training_job,
    verify_job_inputs,
)
from gleipnir.campaign_status import CampaignStatus as Status  # noqa: E402
from gleipnir.qwen35_adapter_rebase import sha256_file  # noqa: E402
from gleipnir.qwen35_fast_training import (  # noqa: E402
    DEFAULT_CAUSAL_CONV1D_TARGET,
    DEFAULT_FLA_TARGET,
    DEFAULT_TRITON_TARGET,
    ensure_qwen35_long_trajectory_kernels,
)

DEFAULT_RESULT_DIR = Path("results/monitoring_objective_ablation")


def gpu_environment(base: dict[str, str], gpu: int) -> dict[str, str]:
    environment = dict(base)
    cache = f"/tmp/gleipnir-monitoring-objective-gpu-{gpu}"
    environment.update(
        CUDA_VISIBLE_DEVICES=str(gpu),
        FLA_DISABLE_BACKEND_DISPATCH="1",
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
        with status.job(name, gpu=gpu):
            run_training_job(
                jobs_path,
                name,
                gpu_environment(environment, gpu),
                allow_non_scaling_job=True,
            )
            validate_training_metadata(
                Path(job["causal_adapter_dir"]) / "training_metadata.json", job
            )


def run(command: list[str], environment: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def run_serving_parity(
    config_path: Path,
    result_dir: Path,
    fast_environment: dict[str, str],
    serving_environment: dict[str, str],
) -> None:
    """Gate ID serving on a bounded causal-master and vLLM comparison."""
    config = json.loads(config_path.read_text())
    job = str(config["model_groups"]["4b"]["parity_job"])
    parity = result_dir / "parity"
    common = ["--config", config_path.as_posix(), "--model-size", "4b"]
    run(
        [
            sys.executable,
            "-m",
            "experiments.tool_trajectory_monitoring.evaluate_distilled_ood_causal",
            *common,
            "--output-root",
            (parity / "eager").as_posix(),
        ],
        fast_environment,
    )
    run(
        [
            sys.executable,
            "-m",
            "experiments.tool_trajectory_monitoring.benchmark_distilled_ood",
            *common,
            "--output-root",
            (parity / "vllm").as_posix(),
            "--only-job",
            job,
            "--include-base",
            "--canary-only",
        ],
        serving_environment,
    )
    run(
        [
            sys.executable,
            "-m",
            "experiments.tool_trajectory_monitoring.compare_distilled_ood_parity",
            "--eager-root",
            (parity / "eager").as_posix(),
            "--vllm-root",
            (parity / "vllm").as_posix(),
            "--model-size",
            "4b",
            "--job-name",
            job,
            "--output",
            (parity / "4b.json").as_posix(),
        ],
        serving_environment,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, default=DEFAULT_RESULT_DIR / "jobs.jsonl")
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
        default=Path("experiments/monitoring_objective_ablation/id_benchmark.json"),
    )
    parser.add_argument("--fla-target", type=Path, default=DEFAULT_FLA_TARGET)
    parser.add_argument(
        "--causal-conv1d-target", type=Path, default=DEFAULT_CAUSAL_CONV1D_TARGET
    )
    parser.add_argument("--triton-target", type=Path, default=DEFAULT_TRITON_TARGET)
    parser.add_argument("--revision", default=os.environ.get("GLEIPNIR_COMMIT"))
    args = parser.parse_args()

    jobs_path = args.jobs.resolve()
    jobs = read_jsonl(jobs_path)
    validate_jobs(jobs)
    verify_job_inputs(jobs)
    evaluation_config = json.loads(args.evaluation_config.read_text())
    validate_evaluation_config(evaluation_config)
    validate_evaluation_inputs(evaluation_config)
    validate_evaluation_jobs(evaluation_config, "4b")
    result_dir = args.result_dir.resolve()
    status = Status(
        args.status.resolve(),
        jobs,
        args.revision,
        metadata={"campaign_id": CAMPAIGN_ID, "strict_ood_consulted": False},
    )
    clean_evaluation_environment = runtime_environment("0")
    try:
        status.update(phase="kernel_preflight")
        fast_environment = ensure_qwen35_long_trajectory_kernels(
            args.fla_target, args.causal_conv1d_target, args.triton_target
        )
        if args.revision:
            fast_environment["GLEIPNIR_COMMIT"] = args.revision
            clean_evaluation_environment["GLEIPNIR_COMMIT"] = args.revision

        selection = args.preflight_selection.resolve()
        preflight_jobs = []
        for kind in ("rationale", "mil"):
            control = next(job for job in jobs if job["kind"] == kind)
            output = result_dir / "preflight" / kind
            preflight_jobs.append(
                {
                    **control,
                    "job_name": f"preflight-{kind}",
                    "train_rows": 1,
                    "max_steps": 1,
                    "num_train_epochs": -1,
                    "gradient_accumulation_steps": 1,
                    "effective_batch_size": 1,
                    "train_sampling_strategy": "sequential",
                    "selection_manifest": selection.as_posix(),
                    "selection_sha256": sha256_file(selection),
                    "selective_torch_compile_canary_tokens": 2048,
                    "save_steps": 1,
                    "output_dir": output.as_posix(),
                    "causal_adapter_dir": (output / "causal_adapter").as_posix(),
                    "model_dir": (output / "model").as_posix(),
                }
            )
        preflight_path = result_dir / "preflight_jobs.jsonl"
        atomic_write_jsonl(preflight_path, preflight_jobs)
        status.update(phase="longest_sequence_preflight", preflight="running")
        for preflight in preflight_jobs:
            run_training_job(
                preflight_path,
                str(preflight["job_name"]),
                gpu_environment(fast_environment, 0),
                preflight=True,
            )
        status.update(phase="training", preflight="passed")

        by_name = {str(job["job_name"]): job for job in jobs}
        lanes = [
            [
                by_name["soft-rationale-w005"],
                by_name["soft-mil-max-w025"],
                by_name["soft-mil-top3-w025"],
            ],
            [by_name["soft-rationale-w020"], by_name["soft-mil-lme-w025"]],
        ]
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    train_lane, lane, gpu, jobs_path, status, fast_environment
                )
                for gpu, lane in enumerate(lanes)
            ]
            for future in futures:
                future.result()

        status.update(phase="serving_parity", active_jobs=[])
        run_serving_parity(
            args.evaluation_config.resolve(),
            result_dir,
            gpu_environment(fast_environment, 0),
            clean_evaluation_environment,
        )
        status.update(phase="id_evaluation", active_jobs=["all-five-objective-arms"])
        run(
            [
                sys.executable,
                "-m",
                "experiments.tool_trajectory_monitoring.benchmark_distilled_ood",
                "--config",
                args.evaluation_config.resolve().as_posix(),
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
                "experiments.monitoring_objective_ablation.summarize",
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
