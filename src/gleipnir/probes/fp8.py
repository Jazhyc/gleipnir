"""Blockwise E4M3 storage for residual-stream activations.

Each token is split into contiguous channel blocks; every block gets one FP32
scale (its absolute maximum over the E4M3 maximum), so residual-stream outlier
channels only coarsen their own block.
"""

from __future__ import annotations

import torch

BLOCK = 128
E4M3_MAX = 448.0


def quantize(
    values: torch.Tensor, block: int = BLOCK
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(uint8 codes [T, H], float32 scales [T, H // block])``."""
    tokens, hidden = values.shape
    if hidden % block:
        raise ValueError(f"hidden size {hidden} is not a multiple of {block}")
    blocks = values.float().reshape(tokens, hidden // block, block)
    scales = blocks.abs().amax(-1) / E4M3_MAX
    scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    codes = (blocks / scales.unsqueeze(-1)).to(torch.float8_e4m3fn)
    return codes.reshape(tokens, hidden).view(torch.uint8), scales


def dequantize(
    codes: torch.Tensor,
    scales: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
    block: int = BLOCK,
) -> torch.Tensor:
    """Invert :func:`quantize` up to E4M3 rounding."""
    tokens, hidden = codes.shape
    values = (
        codes.view(torch.float8_e4m3fn).float().reshape(tokens, hidden // block, block)
    )
    return (values * scales.unsqueeze(-1)).reshape(tokens, hidden).to(dtype)
