#!/usr/bin/env python3
"""Preflight and train the mixed-data Qwen3.5-0.8B adapter on Lambda."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from experiments.tool_trajectory_monitoring.prepare_distillation_scaling import (
    atomic_write_jsonl,
)
from experiments.tool_trajectory_monitoring.prepare_mixed_08b_distillation import (
    GRADIENT_ACCUMULATION_STEPS,
    JOB_NAME,
    MICRO_BATCH_SIZE,
    MODEL_ID,
    MODEL_REVISION,
    validate_mixed_08b_jobs,
)
from experiments.tool_trajectory_monitoring.run_distillation_lambda import (
    Status,
    relocate_jobs,
    run_training_job,
    runtime_environment,
    verify_job_inputs,
)
from gleipnir.qwen35_adapter_rebase import sha256_file
from gleipnir.qwen35_fast_training import DEFAULT_FLA_TARGET, ensure_fla_kernels

DEFAULT_RESULT_DIR = Path("results/tool_trajectory_distillation_mixed_qwen08b")
DEFAULT_SCALING_RESULT_DIR = Path("results/tool_trajectory_distillation_scaling")
PREFLIGHT_SELECTION_SHA256 = (
    "1bdd4d9f99b317119ff7213901ed453f665ff0369e01c68e1a208d72b32925c8"
)


def recipe() -> dict[str, object]:
    return {
        "target": "kimi_soft_only",
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "rank": 128,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "effective_batch_size": 32,
        "max_length": 29_696,
        "quantization": "nf4-double-quant-bf16",
        "direct_logits_mode": "selected_positions",
        "epochs": 1,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, default=DEFAULT_RESULT_DIR / "jobs.jsonl")
    parser.add_argument(
        "--lambda-jobs",
        type=Path,
        default=DEFAULT_RESULT_DIR / "lambda_jobs.jsonl",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/tool_trajectory_monitoring/distillation_scaling"),
    )
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument(
        "--status", type=Path, default=DEFAULT_RESULT_DIR / "lambda_status.json"
    )
    parser.add_argument(
        "--preflight-selection",
        type=Path,
        default=(
            DEFAULT_SCALING_RESULT_DIR / "selections" / "preflight-longest-32.jsonl"
        ),
    )
    parser.add_argument("--fla-target", type=Path, default=DEFAULT_FLA_TARGET)
    parser.add_argument("--revision", default=os.environ.get("GLEIPNIR_COMMIT"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    jobs_path = args.lambda_jobs.resolve()
    jobs = relocate_jobs(
        args.jobs.resolve(),
        jobs_path,
        args.data_dir.resolve(),
        result_dir,
        validator=validate_mixed_08b_jobs,
    )
    verify_job_inputs(jobs)
    status = Status(args.status.resolve(), jobs, args.revision, recipe=recipe())
    environment = runtime_environment()
    try:
        if args.revision:
            environment["GLEIPNIR_COMMIT"] = args.revision
        status.update(phase="kernel_preflight")
        environment.update(ensure_fla_kernels(args.fla_target))

        selection = args.preflight_selection.resolve()
        actual_selection_sha256 = sha256_file(selection)
        if actual_selection_sha256 != PREFLIGHT_SELECTION_SHA256:
            raise ValueError(
                "longest-sequence preflight selection checksum drifted: "
                f"{actual_selection_sha256}"
            )
        full_job = jobs[0]
        preflight_dir = (
            result_dir / "preflight" / "micro1-rank128-longest32"
        ).resolve()
        preflight_job = {
            **full_job,
            "job_name": "preflight-micro1-mixed-qwen35-08b-r128-longest32",
            "design_role": "memory_kernel_and_model_compatibility_preflight",
            "train_rows": 32,
            "max_steps": 1,
            "num_train_epochs": -1,
            "selection_manifest": selection.as_posix(),
            "selection_sha256": actual_selection_sha256,
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
        status.update(phase="training", preflight="passed", active_job=JOB_NAME)
        run_training_job(
            jobs_path,
            JOB_NAME,
            environment,
            allow_non_scaling_job=True,
        )
        status.update(
            state="complete",
            phase="complete",
            active_job=None,
            completed_jobs=[JOB_NAME],
            completed_at_unix=time.time(),
        )
    except BaseException as error:
        status.update(state="failed", error=repr(error), failed_at_unix=time.time())
        raise


if __name__ == "__main__":
    main()
