"""Runtime settings and memory preflight for ordinary packed BF16 LoRA."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from gleipnir.bf16_lora import configure_bf16_reductions, validate_fp4_mlp_lora_config
from gleipnir.packed_sequences import (
    collate_packed_monitoring,
    installed_segmented_sdpa,
    packed_partition,
)
from gleipnir.packed_training_screen import (
    validate_compile_cache_limit,
    validate_learning_tolerance,
)
from gleipnir.training_execution_audit import tensor_digest


def record_packing_canary(
    check: Callable[[], dict[str, Any]],
    metadata: dict[str, Any],
    key: str,
    output: Path,
) -> dict[str, Any]:
    """Preserve a failed numerical receipt before propagating its original error."""
    from gleipnir.packed_training_screen import PackingCanaryError

    receipt = None
    try:
        receipt = check()
        return receipt
    except PackingCanaryError as error:
        receipt = error.receipt
        raise
    finally:
        if receipt is not None:
            metadata[key] = receipt
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")


def validate_packed_training_config(student: Mapping[str, Any]) -> bool:
    """Reject unsupported ordinary packing before loading the model."""
    training = student["training"]
    enabled = training.get("sequence_packing", False)
    if type(enabled) is not bool:
        raise ValueError("sequence_packing must be boolean")
    if not enabled:
        return False
    quantization = student.get("quantization", {})
    fp4_mlp_lora = validate_fp4_mlp_lora_config(student)
    compile_policy = training.get("selective_torch_compile_policy")
    supported_compile_policy = compile_policy == "full_attention_and_linear_shell" or (
        compile_policy == "checkpointed_full_attention_and_linear_shell"
        and training.get("gradient_checkpointing") is True
        and training.get("nonreentrant_checkpointing") is True
    )
    if not (
        (quantization.get("full_bf16_lora") or fp4_mlp_lora)
        and not quantization.get("enabled", True)
        and (quantization.get("mlp_precision") == "bf16" or fp4_mlp_lora)
        and student.get("model_loader") == "causal_lm"
        and student.get("finetuning_mode") == "lora"
        and student.get("attn_implementation") == "sdpa"
        and training.get("gated_delta_backend") == "flashqla"
        and training.get("adaptive_microbatching", {}).get("enabled")
        and not training.get("torch_compile", False)
        and supported_compile_policy
        and training.get("selective_torch_compile_canary_tokens", 0) > 0
        and student.get("lora", {}).get("dropout") == 0
    ):
        raise ValueError("ordinary packing requires the verified BF16 FlashQLA recipe")
    if (
        any(
            training.get(key, 0)
            for key in [
                "completion_loss_weight",
                "pairwise_loss_weight",
                "mil_loss_weight",
                "prefix_loss_weight",
                "ordinal_soft_loss_weight",
            ]
        )
        or training.get("decision_head_mode", "token_logits") != "token_logits"
    ):
        raise ValueError("ordinary packing supports per-example binary LM-head losses")
    validate_compile_cache_limit(training.get("packing_compile_cache_limit", 64))
    tolerance = training.get("packing_learning_gradient_tolerance")
    validate_learning_tolerance(tolerance)
    backend = training.get("packed_attention_backend", "sdpa")
    if (
        tolerance is not None
        and training.get("startup_validation_reference")
        and backend != "flash_attention_4"
    ):
        raise ValueError("packing learning acceptance requires fresh recorded gates")
    if fp4_mlp_lora and training.get("startup_validation_reference"):
        raise ValueError("FP4 MLP packing requires fresh startup validation")
    version = training.get("packed_attention_version")
    if backend not in {"sdpa", "flash_attention_4", "nvidia_mxfp8"} or (
        (backend != "sdpa") != (version is not None)
    ):
        raise ValueError("packed attention requires an explicit supported version")
    if backend == "nvidia_mxfp8" and training.get("startup_validation_reference"):
        raise ValueError("experimental MXFP8 requires fresh model startup validation")
    if backend == "flash_attention_4" and training.get("startup_validation_reference"):
        if tolerance != 0.10 or not training.get("startup_validation_reference_sha256"):
            raise ValueError(
                "FA4 reuse requires a bound receipt; "
                "otherwise use fresh startup validation"
            )
    return True


@contextmanager
def packed_training_runtime(student: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    """Scope the tested attention router, GEMM and compiler settings to training."""
    enabled = validate_packed_training_config(student)
    metadata: dict[str, Any] = {"enabled": enabled}
    if not enabled:
        yield metadata
        return
    from torch._inductor import config as inductor_config

    limit = student["training"].get("packing_compile_cache_limit", 64)
    attention = student["training"].get("packed_attention_backend", "sdpa")
    attention_version = student["training"].get("packed_attention_version")
    backend = torch.backends.cuda.matmul
    previous = (
        backend.allow_bf16_reduced_precision_reduction,
        backend.allow_bf16_reduced_precision_reduction_split_k,
        torch.backends.cuda.preferred_blas_library(),
    )
    try:
        metadata.update(
            bf16_matmul=configure_bf16_reductions(
                allow_reduced_precision=False, allow_split_k=False
            ),
            compile_cache_limit=limit,
            fail_on_recompile_limit_hit=True,
            emulate_precision_casts=True,
            full_attention={
                "sdpa": "segmented_causal_sdpa",
                "flash_attention_4": "varlen_causal_flash_attention_4",
                "nvidia_mxfp8": "segmented_causal_nvidia_mxfp8",
            }[attention],
            attention_backend=attention,
            attention_version=attention_version,
            learning_gradient_tolerance=student["training"].get(
                "packing_learning_gradient_tolerance"
            ),
            convolution="seq_idx",
            recurrent_state="cu_seq_lens",
            positions="reset_per_example",
            max_packed_tokens=student["training"]["adaptive_microbatching"][
                "max_padded_tokens"
            ],
        )
        with (
            installed_segmented_sdpa(attention, attention_version),
            torch._dynamo.config.patch(
                recompile_limit=limit, fail_on_recompile_limit_hit=True
            ),
            inductor_config.patch(emulate_precision_casts=True),
        ):
            yield metadata
    finally:
        backend.allow_bf16_reduced_precision_reduction = previous[:2]
        torch.backends.cuda.preferred_blas_library(previous[2])


def packed_memory_preflight(
    model: torch.nn.Module,
    features: Sequence[dict[str, Any]],
    loss_forward: Callable,
    *,
    token_budget: int,
    logical_batch_size: int,
) -> dict[str, Any]:
    """Backpropagate the largest actual batch without updating master adapters."""
    selected = sorted(features, key=lambda f: -len(f["direct_input_ids"]))[
        :logical_batch_size
    ]
    lengths = [len(f["direct_input_ids"]) for f in selected]
    partition = packed_partition(lengths, token_budget)
    parameters = [p for p in model.parameters() if p.requires_grad]
    initial = tensor_digest(parameters)
    device = parameters[0].device
    was_training = model.training
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        model.train()
        model.zero_grad(set_to_none=True)
        total = 0.0
        for indices in partition:
            loss = loss_forward(
                collate_packed_monitoring([selected[i] for i in indices])
            )
            if not torch.isfinite(loss).item():
                raise FloatingPointError("nonfinite packed preflight loss")
            fraction = len(indices) / len(selected)
            total += float(loss.detach()) * fraction
            (loss * fraction).backward()
        gradients = [p.grad for p in parameters]
        if any(g is None for g in gradients):
            raise FloatingPointError("missing packed preflight adapter gradient")
        norm = float(torch.nn.utils.get_total_norm(gradients, error_if_nonfinite=True))
        if norm <= 0 or tensor_digest(parameters) != initial:
            raise ValueError("packed preflight has zero gradients or changed masters")
        return {
            "passed": True,
            "selection": "longest_actual_training_inputs",
            "examples": len(selected),
            "max_length": max(lengths),
            "physical_sizes": [len(indices) for indices in partition],
            "actual_tokens": sum(lengths),
            "mean_loss": total,
            "gradient_norm": norm,
            "master_unchanged": True,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else 0,
        }
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)
