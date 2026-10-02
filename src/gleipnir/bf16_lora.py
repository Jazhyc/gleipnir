"""Verify ordinary BF16 LoRA without a quantized frozen base."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


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
