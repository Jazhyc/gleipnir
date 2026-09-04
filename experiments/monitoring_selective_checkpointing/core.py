"""Frozen contracts for the selective-checkpointing screen."""

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

JOB_NAME = "linear-attention-checkpointing"
POLICY = "linear_attention_only"
EXPECTED_CHECKPOINTED_LAYERS = [
    index for index in range(32) if (index + 1) % 4 != 0
]


def validate_jobs(jobs: list[dict[str, Any]]) -> None:
    """Reject drift from the single selective-checkpointing candidate."""
    if len(jobs) != 1:
        raise ValueError("expected one selective-checkpointing candidate")
    job = jobs[0]
    expected = {
        "job_name": JOB_NAME,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "seed": SEED,
        "rank": RANK,
        "lora_alpha": LORA_ALPHA,
        "max_length": MAX_LENGTH,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 32,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "gradient_checkpointing": True,
        "gradient_checkpointing_policy": POLICY,
        "trainer_optim": "adamw_torch",
        "learning_rate": LEARNING_RATE,
        "max_steps": OPTIMIZER_STEPS,
        "train_rows": SELECTION_ROWS,
        "student_rows_sha256": STUDENT_ROWS_SHA256,
        "soft_targets_sha256": SOFT_TARGETS_SHA256,
        "require_causal_conv1d": True,
        "triton_version": "3.7.1",
    }
    for key, value in expected.items():
        if job.get(key) != value:
            raise ValueError(
                f"selective-checkpoint job drift for {key}: "
                f"{job.get(key)!r} != {value!r}"
            )


def validate_training_metadata(
    metadata: dict[str, Any], *, expected_steps: int
) -> None:
    """Validate selective layers, fast kernels, and completion."""
    if metadata.get("gradient_checkpointing") is not True:
        raise ValueError("gradient checkpointing was disabled")
    if metadata.get("gradient_checkpointing_policy") != POLICY:
        raise ValueError("selective checkpointing policy drifted")
    if metadata.get("checkpointed_layer_indices") != EXPECTED_CHECKPOINTED_LAYERS:
        raise ValueError("checkpointed decoder layers drifted")
    if metadata.get("materialized_training_inputs") != {
        "completion": False,
        "direct": True,
    }:
        raise ValueError("direct-only input materialization drifted")
    batch = metadata.get("training_batch", {})
    if batch != {
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 32,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
    }:
        raise ValueError("training batch drifted")
    fla = metadata.get("flash_linear_attention", {})
    causal = metadata.get("causal_conv1d", {})
    if (
        fla.get("version") != "0.5.2"
        or fla.get("backend_dispatch_disabled") is not True
        or fla.get("triton_version") != "3.7.1"
        or not metadata.get("gated_delta_kernel_modules")
        or causal.get("version") != "1.6.2.post1"
        or not causal.get("kernel_modules")
    ):
        raise ValueError("fast-kernel metadata is incomplete")
    completed = int(metadata.get("training_state", {}).get("global_step", -1))
    if completed != expected_steps:
        raise ValueError(f"completed {completed} steps instead of {expected_steps}")
    rate = float(
        metadata.get("train_metrics", {}).get("train_samples_per_second", math.nan)
    )
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("invalid throughput")


def summarize(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Compare selective checkpointing with the matched all-layer baseline."""
    validate_training_metadata(candidate, expected_steps=OPTIMIZER_STEPS)
    baseline_rate = float(baseline["train_metrics"]["train_samples_per_second"])
    candidate_rate = float(candidate["train_metrics"]["train_samples_per_second"])
    improvement = candidate_rate / baseline_rate - 1.0
    return {
        "baseline_samples_per_second": baseline_rate,
        "candidate_samples_per_second": candidate_rate,
        "relative_throughput_improvement": improvement,
        "candidate_peak_cuda_memory_gib": float(
            candidate["peak_cuda_memory_allocated_bytes"]
        )
        / 2**30,
        "selected_policy": POLICY if improvement >= 0.05 else "all",
    }
