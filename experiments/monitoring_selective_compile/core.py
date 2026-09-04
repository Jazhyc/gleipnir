"""Frozen contracts for the selective-compilation screen."""

from __future__ import annotations

import math
from typing import Any

from experiments.monitoring_selective_checkpointing.core import (
    EXPECTED_CHECKPOINTED_LAYERS,
)
from experiments.monitoring_selective_checkpointing.core import (
    POLICY as CHECKPOINTING_POLICY,
)
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

JOB_NAME = "uncheckpointed-full-attention-compile"
COMPILE_POLICY = "uncheckpointed_full_attention"
COMPILE_BACKEND = "inductor"
COMPILE_MODE = "default"
EXPECTED_COMPILED_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31]
MAX_UNIQUE_GRAPHS = 16


def validate_jobs(jobs: list[dict[str, Any]]) -> None:
    """Reject drift from the single selective-compilation candidate."""
    if len(jobs) != 1:
        raise ValueError("expected one selective-compilation candidate")
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
        "gradient_checkpointing_policy": CHECKPOINTING_POLICY,
        "selective_torch_compile_policy": COMPILE_POLICY,
        "selective_torch_compile_backend": COMPILE_BACKEND,
        "selective_torch_compile_mode": COMPILE_MODE,
        "selective_torch_compile_dynamic": True,
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
                f"selective-compile job drift for {key}: "
                f"{job.get(key)!r} != {value!r}"
            )


def validate_training_metadata(
    metadata: dict[str, Any], *, expected_steps: int
) -> None:
    """Validate compiler scope, graph count, fast kernels, and completion."""
    if metadata.get("gradient_checkpointing_policy") != CHECKPOINTING_POLICY:
        raise ValueError("selective checkpointing policy drifted")
    if metadata.get("checkpointed_layer_indices") != EXPECTED_CHECKPOINTED_LAYERS:
        raise ValueError("checkpointed decoder layers drifted")
    compiled = metadata.get("selective_torch_compile", {})
    if compiled.get("policy") != COMPILE_POLICY:
        raise ValueError("selective compilation policy drifted")
    if compiled.get("compiled_layer_indices") != EXPECTED_COMPILED_LAYERS:
        raise ValueError("compiled decoder layers drifted")
    if compiled.get("backend") != COMPILE_BACKEND:
        raise ValueError("selective compilation backend drifted")
    if compiled.get("mode") != COMPILE_MODE or compiled.get("dynamic") is not True:
        raise ValueError("selective compilation shape policy drifted")
    if compiled.get("global_torch_compile") is not False:
        raise ValueError("global Torch compilation was enabled")
    unique_graphs = int(
        compiled.get("dynamo_counters", {}).get("stats", {}).get("unique_graphs", 0)
    )
    if not EXPECTED_COMPILED_LAYERS or not 1 <= unique_graphs <= MAX_UNIQUE_GRAPHS:
        raise ValueError(f"unexpected Dynamo graph count: {unique_graphs}")
    if metadata.get("materialized_training_inputs") != {
        "completion": False,
        "direct": True,
    }:
        raise ValueError("direct-only input materialization drifted")
    if metadata.get("training_batch") != {
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


def summarize(
    baseline: dict[str, Any],
    all_layer_baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Compare selective compilation with its eager checkpointing baseline."""
    validate_training_metadata(candidate, expected_steps=OPTIMIZER_STEPS)
    baseline_rate = float(baseline["train_metrics"]["train_samples_per_second"])
    all_layer_rate = float(
        all_layer_baseline["train_metrics"]["train_samples_per_second"]
    )
    candidate_rate = float(candidate["train_metrics"]["train_samples_per_second"])
    improvement = candidate_rate / baseline_rate - 1.0
    combined_improvement = candidate_rate / all_layer_rate - 1.0
    selected = improvement >= 0.05 and combined_improvement >= 0.05
    return {
        "baseline_samples_per_second": baseline_rate,
        "all_layer_baseline_samples_per_second": all_layer_rate,
        "candidate_samples_per_second": candidate_rate,
        "relative_throughput_improvement": improvement,
        "combined_relative_improvement_over_all_layer_baseline": combined_improvement,
        "candidate_peak_cuda_memory_gib": float(
            candidate["peak_cuda_memory_allocated_bytes"]
        )
        / 2**30,
        "dynamo_counters": candidate["selective_torch_compile"]["dynamo_counters"],
        "selected_policy": COMPILE_POLICY if selected else "none",
    }
