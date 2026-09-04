"""Frozen contracts and pure helpers for the monitoring throughput screen."""

from __future__ import annotations

import math
from typing import Any

from experiments.monitoring_lr_sweep.core import (
    LORA_ALPHA,
    MAX_LENGTH,
    MODEL_ID,
    MODEL_REVISION,
    RANK,
    SOFT_TARGETS_SHA256,
    STUDENT_ROWS_SHA256,
)
from gleipnir.monitoring_systems_screen import stable_stratified_selection

SEED = 0
SELECTION_ROWS = 640
OPTIMIZER_STEPS = 20
EFFECTIVE_BATCH_SIZE = 32
LEARNING_RATE = 5e-5
CONDITIONS = (
    {
        "job_name": "mb1-random",
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 32,
        "train_sampling_strategy": "random",
    },
    {
        "job_name": "mb2-random",
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 16,
        "train_sampling_strategy": "random",
    },
    {
        "job_name": "mb2-length-grouped",
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 16,
        "train_sampling_strategy": "group_by_length",
    },
)


def validate_jobs(jobs: list[dict[str, Any]]) -> None:
    """Reject drift from the three matched throughput conditions."""
    if [job.get("job_name") for job in jobs] != [
        condition["job_name"] for condition in CONDITIONS
    ]:
        raise ValueError("throughput condition order drifted")
    for job, condition in zip(jobs, CONDITIONS, strict=True):
        expected = {
            **condition,
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "seed": SEED,
            "rank": RANK,
            "lora_alpha": LORA_ALPHA,
            "max_length": MAX_LENGTH,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "max_steps": OPTIMIZER_STEPS,
            "train_rows": SELECTION_ROWS,
            "student_rows_sha256": STUDENT_ROWS_SHA256,
            "soft_targets_sha256": SOFT_TARGETS_SHA256,
            "require_causal_conv1d": True,
            "fla_disable_backend_dispatch": True,
            "triton_version": "3.7.1",
        }
        for key, value in expected.items():
            if job.get(key) != value:
                raise ValueError(
                    f"job contract drift for {job.get('job_name')!r} {key}: "
                    f"{job.get(key)!r} != {value!r}"
                )
        if (
            int(job["micro_batch_size"]) * int(job["gradient_accumulation_steps"])
            != EFFECTIVE_BATCH_SIZE
        ):
            raise ValueError("effective batch drifted")


def validate_training_metadata(metadata: dict[str, Any], job: dict[str, Any]) -> None:
    """Validate completion and the kernel/batching systems contract."""
    batch = metadata.get("training_batch", {})
    fla = metadata.get("flash_linear_attention", {})
    causal = metadata.get("causal_conv1d", {})
    batching = metadata.get("batching", {})
    timing = metadata.get("optimizer_step_timing", {})
    if batch != {
        "micro_batch_size": job["micro_batch_size"],
        "gradient_accumulation_steps": job["gradient_accumulation_steps"],
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
    }:
        raise ValueError("training batch metadata drifted")
    if (
        fla.get("available") is not True
        or fla.get("required") is not True
        or fla.get("version") != "0.5.2"
        or fla.get("backend_dispatch_disabled") is not True
        or fla.get("triton_version") != "3.7.1"
        or not metadata.get("gated_delta_kernel_modules")
    ):
        raise ValueError("pinned FLA metadata is incomplete")
    if (
        causal.get("available") is not True
        or causal.get("required") is not True
        or causal.get("version") != "1.6.2.post1"
        or not causal.get("kernel_modules")
    ):
        raise ValueError("pinned causal-conv1d metadata is incomplete")
    if batching.get("train_sampling_strategy") != job["train_sampling_strategy"]:
        raise ValueError("sampling strategy metadata drifted")
    completed_steps = int(metadata.get("training_state", {}).get("global_step", -1))
    if completed_steps != OPTIMIZER_STEPS:
        raise ValueError("throughput run did not complete exactly 20 steps")
    if int(timing.get("recorded_steps", -1)) != OPTIMIZER_STEPS:
        raise ValueError("optimizer step timing is incomplete")
    for key in ("steady_mean_seconds", "steady_median_seconds"):
        value = float(timing.get(key, math.nan))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid optimizer timing {key}")
    samples_per_second = float(
        metadata.get("train_metrics", {}).get("train_samples_per_second", math.nan)
    )
    if not math.isfinite(samples_per_second) or samples_per_second <= 0:
        raise ValueError("invalid training throughput")


def summarize_condition(
    metadata: dict[str, Any], job: dict[str, Any]
) -> dict[str, Any]:
    """Extract comparable systems metrics from one validated condition."""
    validate_training_metadata(metadata, job)
    metrics = metadata["train_metrics"]
    timing = metadata["optimizer_step_timing"]
    padding = metadata["batching"]["padding"]
    runtime = float(metrics["train_runtime"])
    return {
        "job_name": job["job_name"],
        "micro_batch_size": job["micro_batch_size"],
        "gradient_accumulation_steps": job["gradient_accumulation_steps"],
        "train_sampling_strategy": job["train_sampling_strategy"],
        "train_runtime_seconds": runtime,
        "train_samples_per_second": float(metrics["train_samples_per_second"]),
        "unpadded_direct_tokens_per_second": float(padding["direct_tokens"]) / runtime,
        "direct_padding_fraction": float(padding["direct_padding_fraction"] or 0.0),
        "steady_mean_step_seconds": float(timing["steady_mean_seconds"]),
        "steady_median_step_seconds": float(timing["steady_median_seconds"]),
        "projected_full_cell_hours": float(timing["steady_mean_seconds"]) * 272 / 3600,
        "peak_cuda_memory_allocated_gib": float(
            metadata["peak_cuda_memory_allocated_bytes"]
        )
        / 2**30,
    }


def select_recipe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Select a material throughput improvement over random microbatch 2."""
    if {row["job_name"] for row in rows} != {
        condition["job_name"] for condition in CONDITIONS
    }:
        raise ValueError("summary is missing throughput conditions")
    baseline = next(row for row in rows if row["job_name"] == "mb2-random")
    baseline_rate = float(baseline["train_samples_per_second"])
    fastest = max(rows, key=lambda row: float(row["train_samples_per_second"]))
    fastest_rate = float(fastest["train_samples_per_second"])
    contenders = [
        row
        for row in rows
        if float(row["train_samples_per_second"]) >= fastest_rate * 0.98
    ]
    best = min(
        contenders,
        key=lambda row: (
            float(row["direct_padding_fraction"]),
            int(row["micro_batch_size"]),
        ),
    )
    improvement = float(best["train_samples_per_second"]) / baseline_rate - 1.0
    selected = best if improvement >= 0.05 else baseline
    return {
        "baseline_job": baseline["job_name"],
        "fastest_measured_job": fastest["job_name"],
        "best_with_tie_breaks_job": best["job_name"],
        "best_relative_improvement": improvement,
        "selected_job": selected["job_name"],
        "baseline_retained": selected is baseline,
    }
