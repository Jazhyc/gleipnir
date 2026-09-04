"""Frozen contracts for the eight-step partial-checkpointing screen."""

from __future__ import annotations

import math
from typing import Any

from experiments.monitoring_selective_checkpointing.core import (
    EXPECTED_CHECKPOINTED_LAYERS,
)
from experiments.monitoring_training_throughput.core import (
    EFFECTIVE_BATCH_SIZE,
    LEARNING_RATE,
    LORA_ALPHA,
    MAX_LENGTH,
    MODEL_ID,
    MODEL_REVISION,
    RANK,
    SEED,
    SELECTION_ROWS,
    SOFT_TARGETS_SHA256,
    STUDENT_ROWS_SHA256,
)

CONTROL_NAME = "all-linear-checkpoint-control"
CANDIDATE_NAME = "checkpoint-four-linear-candidate"
COMPILE_POLICY = "full_attention_and_linear_shell"
CONTROL_CHECKPOINT_POLICY = "linear_attention_only"
CANDIDATE_CHECKPOINT_POLICY = "explicit"
CANDIDATE_CHECKPOINTED_LAYERS = [2, 10, 18, 26]
EXPECTED_ALL_LAYERS = list(range(32))
OPTIMIZER_STEPS = 8
MAX_UNIQUE_GRAPHS = 16
MAX_PREFLIGHT_ALLOCATED_GIB = 72.0
MINIMUM_RELATIVE_IMPROVEMENT = 0.01


def expected_checkpoint_layers(policy: str) -> list[int]:
    """Return the frozen checkpoint set for one condition."""
    if policy == CONTROL_CHECKPOINT_POLICY:
        return EXPECTED_CHECKPOINTED_LAYERS
    if policy == CANDIDATE_CHECKPOINT_POLICY:
        return CANDIDATE_CHECKPOINTED_LAYERS
    raise ValueError(f"unexpected checkpointing policy: {policy}")


def validate_jobs(jobs: list[dict[str, Any]]) -> None:
    """Reject drift from the two matched eight-step conditions."""
    if [job.get("job_name") for job in jobs] != [CONTROL_NAME, CANDIDATE_NAME]:
        raise ValueError("partial-checkpointing jobs or order drifted")
    policies = {
        CONTROL_NAME: CONTROL_CHECKPOINT_POLICY,
        CANDIDATE_NAME: CANDIDATE_CHECKPOINT_POLICY,
    }
    for job in jobs:
        policy = policies[job["job_name"]]
        explicit = (
            CANDIDATE_CHECKPOINTED_LAYERS
            if policy == CANDIDATE_CHECKPOINT_POLICY
            else None
        )
        expected = {
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
            "gradient_checkpointing_policy": policy,
            "gradient_checkpointing_layer_indices": explicit,
            "selective_torch_compile_policy": COMPILE_POLICY,
            "selective_torch_compile_backend": "inductor",
            "selective_torch_compile_mode": "default",
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
                    f"partial-checkpointing drift for {key}: "
                    f"{job.get(key)!r} != {value!r}"
                )


def validate_training_metadata(
    metadata: dict[str, Any],
    *,
    checkpoint_policy: str,
    expected_steps: int,
    enforce_memory_ceiling: bool = False,
) -> None:
    """Validate checkpoint set, compile boundaries, kernels, and completion."""
    expected_layers = expected_checkpoint_layers(checkpoint_policy)
    if metadata.get("gradient_checkpointing_policy") != checkpoint_policy:
        raise ValueError("checkpointing policy drifted")
    if metadata.get("checkpointed_layer_indices") != expected_layers:
        raise ValueError("checkpointed layers drifted")
    expected_requested = (
        CANDIDATE_CHECKPOINTED_LAYERS
        if checkpoint_policy == CANDIDATE_CHECKPOINT_POLICY
        else None
    )
    if metadata.get("gradient_checkpointing_layer_indices") != expected_requested:
        raise ValueError("requested checkpoint layers drifted")
    compiled = metadata.get("selective_torch_compile", {})
    if compiled.get("policy") != COMPILE_POLICY:
        raise ValueError("compile policy drifted")
    if compiled.get("compiled_layer_indices") != EXPECTED_ALL_LAYERS:
        raise ValueError("compiled layer set drifted")
    if (
        compiled.get("disabled_linear_attention_layer_indices")
        != EXPECTED_CHECKPOINTED_LAYERS
    ):
        raise ValueError("eager linear-attention boundary drifted")
    if (
        compiled.get("backend") != "inductor"
        or compiled.get("mode") != "default"
        or compiled.get("dynamic") is not True
        or compiled.get("global_torch_compile") is not False
    ):
        raise ValueError("compiler configuration drifted")
    unique_graphs = int(
        compiled.get("dynamo_counters", {}).get("stats", {}).get("unique_graphs", 0)
    )
    if not 1 <= unique_graphs <= MAX_UNIQUE_GRAPHS:
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
    peak_gib = float(metadata.get("peak_cuda_memory_allocated_bytes", math.inf)) / 2**30
    if enforce_memory_ceiling and peak_gib > MAX_PREFLIGHT_ALLOCATED_GIB:
        raise ValueError(
            f"preflight allocated {peak_gib:.2f} GiB above "
            f"{MAX_PREFLIGHT_ALLOCATED_GIB:.2f} GiB ceiling"
        )


def summarize(jobs: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare partial checkpointing with the current compiled control."""
    validate_jobs(jobs)
    conditions: dict[str, dict[str, Any]] = {}
    for job in jobs:
        metadata = job["training_metadata"]
        validate_training_metadata(
            metadata,
            checkpoint_policy=job["gradient_checkpointing_policy"],
            expected_steps=OPTIMIZER_STEPS,
        )
        timing = metadata["optimizer_step_timing"]
        conditions[job["job_name"]] = {
            "samples_per_second": float(
                metadata["train_metrics"]["train_samples_per_second"]
            ),
            "train_runtime_seconds": float(metadata["train_metrics"]["train_runtime"]),
            "steady_mean_step_seconds": float(timing["steady_mean_seconds"]),
            "peak_cuda_memory_gib": float(
                metadata["peak_cuda_memory_allocated_bytes"]
            )
            / 2**30,
            "unique_graphs": int(
                metadata["selective_torch_compile"]["dynamo_counters"]["stats"][
                    "unique_graphs"
                ]
            ),
        }
    control_rate = conditions[CONTROL_NAME]["samples_per_second"]
    candidate_rate = conditions[CANDIDATE_NAME]["samples_per_second"]
    improvement = candidate_rate / control_rate - 1.0
    selected = (
        CANDIDATE_CHECKPOINT_POLICY
        if improvement >= MINIMUM_RELATIVE_IMPROVEMENT
        else CONTROL_CHECKPOINT_POLICY
    )
    return {
        "conditions": conditions,
        "relative_throughput_improvement": improvement,
        "selected_checkpointing_policy": selected,
        "selected_checkpointed_layer_indices": (
            CANDIDATE_CHECKPOINTED_LAYERS
            if selected == CANDIDATE_CHECKPOINT_POLICY
            else EXPECTED_CHECKPOINTED_LAYERS
        ),
    }
