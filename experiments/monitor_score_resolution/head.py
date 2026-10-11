"""FP32 projection from unchanged BF16 hidden states and decision weight rows."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def project(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute before output rounding, not by widening already-rounded logits."""
    if hidden.dtype != torch.bfloat16 or weight.dtype != torch.float32:
        raise ValueError("expected BF16 hidden states and promoted FP32 head rows")
    if hidden.shape[-1] != weight.shape[-1] or weight.shape[0] != 2:
        raise ValueError("two-row head geometry changed")
    return F.linear(hidden.float(), weight)
