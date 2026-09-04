"""Frozen contracts for the eight-step linear-shell compile screen."""

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

CONTROL_NAME = "full-attention-compile-control"
CANDIDATE_NAME = "full-attention-plus-linear-shell-compile"
CONTROL_POLICY = "uncheckpointed_full_attention"
CANDIDATE_POLICY = "full_attention_and_linear_shell"
OPTIMIZER_STEPS = 8
EXPECTED_FULL_ATTENTION_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31]
EXPECTED_ALL_LAYERS = list(range(32))
MAX_UNIQUE_GRAPHS = 16


def validate_jobs(jobs: list[dict[str, Any]]) -> None:
    """Reject drift from the two matched eight-step compile conditions."""
    if [job.get("job_name") for job in jobs] != [CONTROL_NAME, CANDIDATE_NAME]:
        raise ValueError("linear-shell jobs or order drifted")
    policies = {
        CONTROL_NAME: CONTROL_POLICY,
        CANDIDATE_NAME: CANDIDATE_POLICY,
    }
    for job in jobs:
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
            "gradient_checkpointing_policy": "linear_attention_only",
            "selective_torch_compile_policy": policies[job["job_name"]],
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
                    f"linear-shell job drift for {key}: "
                    f"{job.get(key)!r} != {value!r}"
                )


def validate_training_metadata(
    metadata: dict[str, Any],
    *,
    policy: str,
    expected_steps: int,
    require_canary: bool = False,
) -> None:
    """Validate compile boundaries, kernels, graph count, and completion."""
    if metadata.get("gradient_checkpointing_policy") != "linear_attention_only":
        raise ValueError("checkpointing policy drifted")
    if metadata.get("checkpointed_layer_indices") != EXPECTED_CHECKPOINTED_LAYERS:
        raise ValueError("checkpointed layers drifted")
    compiled = metadata.get("selective_torch_compile", {})
    if compiled.get("policy") != policy:
        raise ValueError("compile policy drifted")
    expected_compiled = (
        EXPECTED_ALL_LAYERS
        if policy == CANDIDATE_POLICY
        else EXPECTED_FULL_ATTENTION_LAYERS
    )
    expected_disabled = (
        EXPECTED_CHECKPOINTED_LAYERS if policy == CANDIDATE_POLICY else []
    )
    if compiled.get("compiled_layer_indices") != expected_compiled:
        raise ValueError("compiled layer set drifted")
    if compiled.get("disabled_linear_attention_layer_indices", []) != expected_disabled:
        raise ValueError("eager linear-attention boundary drifted")
    if (
        compiled.get("backend") != "inductor"
        or compiled.get("mode") != "default"
        or compiled.get("dynamic") is not True
        or compiled.get("global_torch_compile") is not False
    ):
        raise ValueError("compiler configuration drifted")
    if require_canary and compiled.get("canary", {}).get("passed") is not True:
        raise ValueError("same-weights compile canary did not pass")
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


def summarize(jobs: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare the candidate linear shell with the compiled control."""
    validate_jobs(jobs)
    conditions = {}
    for job in jobs:
        metadata = job["training_metadata"]
        policy = job["selective_torch_compile_policy"]
        validate_training_metadata(
            metadata, policy=policy, expected_steps=OPTIMIZER_STEPS
        )
        name = job["job_name"]
        timing = metadata["optimizer_step_timing"]
        conditions[name] = {
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
    return {
        "conditions": conditions,
        "relative_throughput_improvement": improvement,
        "selected_policy": CANDIDATE_POLICY if improvement >= 0.01 else CONTROL_POLICY,
    }
