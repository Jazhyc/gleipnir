"""Frozen contracts for the gradient-checkpointing throughput screen."""

from __future__ import annotations

import math
from typing import Any

from experiments.monitoring_training_throughput.core import (
    EFFECTIVE_BATCH_SIZE,
    LEARNING_RATE,
    LORA_ALPHA,
    MAX_LENGTH,
    MODEL_ID,
    MODEL_REVISION,
    OPTIMIZER_STEPS,
    RANK,
    SEED,
    SELECTION_ROWS,
    SOFT_TARGETS_SHA256,
    STUDENT_ROWS_SHA256,
)

JOB_NAME = "mb1-no-gradient-checkpointing"
MICRO_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 32
MINIMUM_RELATIVE_IMPROVEMENT = 0.05


def validate_jobs(jobs: list[dict[str, Any]]) -> None:
    """Reject drift from the single checkpointing intervention."""
    if len(jobs) != 1:
        raise ValueError("expected exactly one checkpointing intervention")
    job = jobs[0]
    expected = {
        "job_name": JOB_NAME,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "seed": SEED,
        "rank": RANK,
        "lora_alpha": LORA_ALPHA,
        "max_length": MAX_LENGTH,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "max_steps": OPTIMIZER_STEPS,
        "train_rows": SELECTION_ROWS,
        "train_sampling_strategy": "random",
        "gradient_checkpointing": False,
        "student_rows_sha256": STUDENT_ROWS_SHA256,
        "soft_targets_sha256": SOFT_TARGETS_SHA256,
        "require_causal_conv1d": True,
        "fla_disable_backend_dispatch": True,
        "triton_version": "3.7.1",
    }
    for key, value in expected.items():
        if job.get(key) != value:
            raise ValueError(
                f"checkpointing job contract drift for {key}: "
                f"{job.get(key)!r} != {value!r}"
            )


def validate_training_metadata(
    metadata: dict[str, Any], *, expected_steps: int
) -> None:
    """Validate the memory, kernel, and completion contract."""
    batch = metadata.get("training_batch", {})
    fla = metadata.get("flash_linear_attention", {})
    causal = metadata.get("causal_conv1d", {})
    if batch != {
        "micro_batch_size": MICRO_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
    }:
        raise ValueError("training batch metadata drifted")
    if metadata.get("gradient_checkpointing") is not False:
        raise ValueError("gradient checkpointing was not disabled")
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
    completed = int(metadata.get("training_state", {}).get("global_step", -1))
    if completed != expected_steps:
        raise ValueError(f"completed {completed} steps instead of {expected_steps}")
    rate = float(
        metadata.get("train_metrics", {}).get("train_samples_per_second", math.nan)
    )
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("invalid training throughput")


def summarize(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Compare the candidate with the completed checkpointed baseline."""
    validate_training_metadata(candidate, expected_steps=OPTIMIZER_STEPS)
    if baseline.get("gradient_checkpointing", True) is not True:
        raise ValueError("baseline did not use gradient checkpointing")
    baseline_batch = baseline.get("training_batch", {})
    if baseline_batch.get("effective_batch_size") != EFFECTIVE_BATCH_SIZE:
        raise ValueError("baseline effective batch drifted")
    baseline_rate = float(baseline["train_metrics"]["train_samples_per_second"])
    candidate_rate = float(candidate["train_metrics"]["train_samples_per_second"])
    improvement = candidate_rate / baseline_rate - 1.0
    padding = candidate["batching"]["padding"]
    runtime = float(candidate["train_metrics"]["train_runtime"])
    return {
        "baseline": {
            "gradient_checkpointing": True,
            "train_samples_per_second": baseline_rate,
            "peak_cuda_memory_allocated_gib": float(
                baseline["peak_cuda_memory_allocated_bytes"]
            )
            / 2**30,
        },
        "candidate": {
            "gradient_checkpointing": False,
            "train_samples_per_second": candidate_rate,
            "unpadded_direct_tokens_per_second": float(padding["direct_tokens"])
            / runtime,
            "steady_mean_step_seconds": float(
                candidate["optimizer_step_timing"]["steady_mean_seconds"]
            ),
            "peak_cuda_memory_allocated_gib": float(
                candidate["peak_cuda_memory_allocated_bytes"]
            )
            / 2**30,
        },
        "relative_throughput_improvement": improvement,
        "selected_gradient_checkpointing": not (
            improvement >= MINIMUM_RELATIVE_IMPROVEMENT
        ),
    }
