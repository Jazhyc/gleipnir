"""Frozen contracts for the AdamW implementation throughput screen."""

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

CONDITIONS = (
    {"job_name": "adamw-torch", "trainer_optim": "adamw_torch"},
    {
        "job_name": "adamw-torch-fused",
        "trainer_optim": "adamw_torch_fused",
    },
)
MINIMUM_RELATIVE_IMPROVEMENT = 0.03


def validate_jobs(jobs: list[dict[str, Any]]) -> None:
    """Reject drift from the matched two-condition screen."""
    if [job.get("job_name") for job in jobs] != [
        condition["job_name"] for condition in CONDITIONS
    ]:
        raise ValueError("AdamW condition order drifted")
    for job, condition in zip(jobs, CONDITIONS, strict=True):
        expected = {
            **condition,
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
                    f"AdamW job contract drift for {job.get('job_name')} "
                    f"{key}: {job.get(key)!r} != {value!r}"
                )


def validate_training_metadata(
    metadata: dict[str, Any], job: dict[str, Any], *, expected_steps: int
) -> None:
    """Validate completion and the systems contract."""
    batch = metadata.get("training_batch", {})
    if batch != {
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 32,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
    }:
        raise ValueError("training batch metadata drifted")
    if metadata.get("gradient_checkpointing") is not True:
        raise ValueError("gradient checkpointing drifted")
    if metadata.get("materialized_training_inputs") != {
        "completion": False,
        "direct": True,
    }:
        raise ValueError("unused completion tensors were materialized")
    if metadata.get("optimization", {}).get("trainer_optim") != job["trainer_optim"]:
        raise ValueError("Trainer optimizer implementation drifted")
    fla = metadata.get("flash_linear_attention", {})
    causal = metadata.get("causal_conv1d", {})
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


def summarize_condition(
    metadata: dict[str, Any], job: dict[str, Any]
) -> dict[str, Any]:
    """Extract comparable metrics from one condition."""
    validate_training_metadata(metadata, job, expected_steps=OPTIMIZER_STEPS)
    runtime = float(metadata["train_metrics"]["train_runtime"])
    padding = metadata["batching"]["padding"]
    return {
        "job_name": job["job_name"],
        "trainer_optim": job["trainer_optim"],
        "train_runtime_seconds": runtime,
        "train_samples_per_second": float(
            metadata["train_metrics"]["train_samples_per_second"]
        ),
        "unpadded_direct_tokens_per_second": float(padding["direct_tokens"])
        / runtime,
        "steady_mean_step_seconds": float(
            metadata["optimizer_step_timing"]["steady_mean_seconds"]
        ),
        "peak_cuda_memory_allocated_gib": float(
            metadata["peak_cuda_memory_allocated_bytes"]
        )
        / 2**30,
    }


def select_recipe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Select fused AdamW only for a material throughput gain."""
    by_name = {row["job_name"]: row for row in rows}
    if set(by_name) != {condition["job_name"] for condition in CONDITIONS}:
        raise ValueError("summary is missing an AdamW condition")
    baseline = by_name["adamw-torch"]
    fused = by_name["adamw-torch-fused"]
    improvement = (
        float(fused["train_samples_per_second"])
        / float(baseline["train_samples_per_second"])
        - 1.0
    )
    return {
        "relative_throughput_improvement": improvement,
        "selected_job": (
            "adamw-torch-fused"
            if improvement >= MINIMUM_RELATIVE_IMPROVEMENT
            else "adamw-torch"
        ),
    }
