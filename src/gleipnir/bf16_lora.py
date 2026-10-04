"""Verify ordinary BF16 LoRA without a quantized frozen base."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def configure_bf16_reductions(
    *,
    allow_reduced_precision: bool | None = None,
    allow_split_k: bool | None = None,
) -> dict[str, bool | str]:
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
        if not selected_split_k:
            # Torch enforces this at the first CUDA GEMM, rather than at the setter.
            torch.backends.cuda.preferred_blas_library("cublaslt")
        backend.allow_bf16_reduced_precision_reduction = (
            selected_reduction,
            selected_split_k,
        )
    return {
        "blas_library": str(torch.backends.cuda.preferred_blas_library())
        .split(".")[-1]
        .lower(),
        "allow_reduced_precision": backend.allow_bf16_reduced_precision_reduction,
        "allow_split_k": backend.allow_bf16_reduced_precision_reduction_split_k,
    }


def _original_base_metadata(model: nn.Module) -> dict[str, Any]:
    """Verify original BF16 parameters and FP32 masters, excluding k-bit loading."""
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


def bf16_lora_metadata(model: nn.Module) -> dict[str, Any]:
    """Fail closed on native quantized bases or non-FP32 master adapters."""
    if any(
        hasattr(module, "runtime") and hasattr(module.runtime, "activation_config")
        for module in model.modules()
    ):
        raise ValueError("ordinary BF16 LoRA must contain no native quantized modules")
    return _original_base_metadata(model)


def fp4_mlp_lora_metadata(model: nn.Module) -> dict[str, Any]:
    """Verify native MLP-only W4A4 with original BF16 bases and FP32 LoRA masters."""
    from gleipnir.fouroversix_training import FrozenFourOverSixLinear, is_mlp_base

    original = _original_base_metadata(model)
    native = [
        (name, module)
        for name, module in model.named_modules(remove_duplicate=False)
        if isinstance(module, FrozenFourOverSixLinear)
    ]
    mlps = [
        (name, module)
        for name, module in model.named_modules()
        if is_mlp_base(name) and not hasattr(module, "base_layer")
    ]
    if (
        len(native) != 96
        or len(mlps) != 96
        or any(not is_mlp_base(name) for name, _ in native)
        or any(not isinstance(module, FrozenFourOverSixLinear) for _, module in mlps)
    ):
        raise ValueError("FP4 MLP LoRA requires exactly 96 native decoder MLP bases")
    if any(
        not module.runtime.row_scaled_activations
        or module.runtime.dequantized_weight is None
        or module.runtime.dequantized_weight.dtype != torch.bfloat16
        or module.runtime.activation_selector != "strict"
        for _, module in native
    ):
        raise ValueError(
            "FP4 MLP LoRA requires per-token strict FP4 and decoded BF16 backward"
        )
    return {
        **original,
        "frozen_dtype": "mixed_native_fp4_mlp_bf16_other",
        "original_weight_dtype": "torch.bfloat16",
        "mlp_forward": "native_cutlass_w4a4",
        "base_input_gradient": "decoded_forward_weight_bf16",
        "native_mlp_modules": [name for name, _ in native],
        "quantized_modules": [name for name, _ in native],
        "non_mlp_base_storage": "bf16",
        "reference_weights_retained": True,
    }


def validate_fp4_mlp_lora_config(student: dict[str, Any]) -> bool:
    """Permit explicit mixed LoRA and preserve historical QLoRA loading."""
    quantization = student.get("quantization", {})
    enabled = quantization.get("fp4_mlp_lora", False)
    if type(enabled) is not bool:
        raise ValueError("fp4_mlp_lora must be boolean")
    if enabled and not (
        quantization.get("mlp_precision") == "fouroversix"
        and not quantization.get("enabled", True)
        and not quantization.get("full_bf16_lora", False)
        and quantization.get("fp4_backward_mode") == "dequantized_bf16"
        and quantization.get("fp4_row_scaled_activations") is True
        and quantization.get("fp4_activation_selector", "strict") == "strict"
        and student.get("model_loader") == "causal_lm"
        and student.get("finetuning_mode") == "lora"
    ):
        raise ValueError(
            "FP4 MLP LoRA requires unquantized causal loading, "
            "per-token FP4 and BF16 backward"
        )
    return enabled
