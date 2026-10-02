"""Verify ordinary BF16 LoRA without a quantized frozen base."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def configure_bf16_reductions(
    *,
    allow_reduced_precision: bool | None = None,
    allow_split_k: bool | None = None,
) -> dict[str, bool]:
    """Record and optionally constrain GEMM reductions without changing storage."""
    if any(
        value is not None and type(value) is not bool
        for value in [allow_reduced_precision, allow_split_k]
    ):
        raise ValueError("BF16 reduction controls must be booleans")
    backend = torch.backends.cuda.matmul
    reduction = backend.allow_bf16_reduced_precision_reduction
    split_k = backend.allow_bf16_reduced_precision_reduction_split_k
    if allow_reduced_precision is not None or allow_split_k is not None:
        selected_reduction = (
            reduction if allow_reduced_precision is None else allow_reduced_precision
        )
        selected_split_k = split_k if allow_split_k is None else allow_split_k
        if selected_reduction and not selected_split_k:
            raise ValueError("disabling split-K requires disabled precision reductions")
        backend.allow_bf16_reduced_precision_reduction = (
            selected_reduction,
            selected_split_k,
        )
    return {
        "allow_reduced_precision": backend.allow_bf16_reduced_precision_reduction,
        "allow_split_k": backend.allow_bf16_reduced_precision_reduction_split_k,
    }


def bf16_lora_metadata(model: nn.Module) -> dict[str, Any]:
    """Fail closed on quantized bases or non-FP32 master adapters."""
    quantized = [
        name
        for name, module in model.named_modules()
        if module.__class__.__name__ in {"Linear4bit", "Linear8bitLt"}
        or module.__class__.__module__.startswith("bitsandbytes.")
    ]
    if quantized or getattr(model, "is_loaded_in_4bit", False):
        raise ValueError("ordinary BF16 LoRA must contain no quantized modules")
    frozen_elements = 0
    trainable_elements = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if "lora_" not in name or parameter.dtype != torch.float32:
                raise ValueError("ordinary BF16 LoRA requires FP32 master adapters")
            trainable_elements += parameter.numel()
        else:
            if parameter.dtype != torch.bfloat16:
                raise ValueError(
                    f"ordinary BF16 LoRA requires BF16 frozen bases: {name}"
                )
            frozen_elements += parameter.numel()
    if not frozen_elements or not trainable_elements:
        raise ValueError("ordinary BF16 LoRA requires frozen bases and adapters")
    return {
        "verified": True,
        "frozen_dtype": "torch.bfloat16",
        "frozen_elements": frozen_elements,
        "master_dtype": "torch.float32",
        "trainable_elements": trainable_elements,
        "quantized_modules": quantized,
    }
