"""Explicit bounded memory-policy overrides for precision throughput screens."""

from __future__ import annotations

from typing import Any


def validate_memory_recipe(config: dict[str, Any]) -> None:
    """Validate bounded BF16 checkpoint and native FP4 memory interventions."""
    enabled = config.get("reference_weights_on_cpu", False)
    indices = config.get("checkpoint_layer_indices")
    budget = config.get("adaptive_token_budget")
    bounded_bf16 = (
        config.get("full_bf16_lora", False)
        and config.get("ten_step_learning_comparison", False)
        and config.get("conditions") == ["bf16"]
        and config.get("steps") == 10
        and not config.get("diagnostics_only", False)
        and not enabled
    )
    disable_checkpointing = config.get("disable_gradient_checkpointing", False)
    if type(disable_checkpointing) is not bool:
        raise ValueError("disable_gradient_checkpointing must be boolean")
    if disable_checkpointing and not (
        bounded_bf16 and indices is None and budget is None
    ):
        raise ValueError("disabling checkpointing requires the bounded BF16 comparison")
    if (indices is not None or budget is not None) and not (
        enabled or (bounded_bf16 and indices is None)
    ):
        raise ValueError("memory-policy overrides require reference-weight offload")
    if enabled and not (
        config["conditions"] == ["fouroversix"]
        and config["backward_mode"] == "dequantized_bf16"
        and not config.get("capture_native_operands", False)
    ):
        raise ValueError(
            "reference offload requires native FP4 and cached BF16 backward"
        )
    if indices is not None and (
        not isinstance(indices, list)
        or not indices
        or any(type(index) is not int or not 0 <= index < 32 for index in indices)
        or indices != sorted(set(indices))
    ):
        raise ValueError("checkpoint override requires sorted unique Qwen4B layers")
    allowed_budgets = {16384, 24576, 32768} if bounded_bf16 else {16384, 24576}
    if budget is not None and (
        type(budget) is not int or budget not in allowed_budgets
    ):
        raise ValueError("adaptive budget must be a predeclared bounded token count")


def apply_memory_recipe(job: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Keep sources and effective batch intact while recording physical changes."""
    validate_memory_recipe(config)
    result = dict(job)
    if config.get("disable_gradient_checkpointing", False):
        result["gradient_checkpointing"] = False
        result["gradient_checkpointing_policy"] = "all"
        result["gradient_checkpointing_layer_indices"] = None
    if config.get("checkpoint_layer_indices") is not None:
        result["gradient_checkpointing"] = True
        result["gradient_checkpointing_policy"] = "explicit"
        result["gradient_checkpointing_layer_indices"] = list(
            config["checkpoint_layer_indices"]
        )
    if config.get("adaptive_token_budget") is not None:
        result["adaptive_microbatching"] = {
            **job["adaptive_microbatching"],
            "max_padded_tokens": config["adaptive_token_budget"],
        }
    return result
