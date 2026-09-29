"""Experimental Triton NVFP4 packing with explicit group clipping.

Group scales are E4M3, values are nearest-even E2M1, and adjacent values
occupy the low then high nibble. Native GEMM remains the vLLM FP4 kernel.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _pack(
    X,
    INV,
    PACKED,
    SCALES,
    K_GROUPS: tl.constexpr,
    GROUPS,
    K_PADDED: tl.constexpr,
    CLIP: tl.constexpr,
    SWIZZLED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    group = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(
        X + group[:, None] * 16 + tl.arange(0, 16)[None, :],
        group[:, None] < GROUPS,
        other=0,
    ).to(tl.float32)
    inv = tl.load(INV)
    maximum = tl.max(tl.abs(values), axis=1)
    scales = tl.minimum(maximum * (CLIP / 6.0) * inv, 448.0)
    rounded = scales.to(tl.float8e4nv).to(tl.float32)
    normalized = tl.abs(values) * inv / tl.where(rounded > 0, rounded, 1.0)[:, None]
    # Midpoints alternate strict/non-strict comparisons for ties-to-even.
    code = (
        (normalized > 0.25).to(tl.int32)
        + (normalized >= 0.75).to(tl.int32)
        + (normalized > 1.25).to(tl.int32)
        + (normalized >= 1.75).to(tl.int32)
        + (normalized > 2.5).to(tl.int32)
        + (normalized >= 3.5).to(tl.int32)
        + (normalized > 5.0).to(tl.int32)
    )
    sign = (values.to(tl.int32, bitcast=True) >> 28) & 8
    code = (code | sign).to(tl.uint8)
    low, high = tl.split(tl.reshape(code, (BLOCK, 8, 2)))
    packed = low | (high << 4)
    tl.store(
        PACKED + group[:, None] * 8 + tl.arange(0, 8)[None, :],
        packed,
        group[:, None] < GROUPS,
    )
    if SWIZZLED:
        row, col = group // K_GROUPS, group % K_GROUPS
        offset = (
            ((row // 128 * (K_PADDED // 4) + col // 4) * 32 + row % 32) * 4
            + (row % 128) // 32
        ) * 4 + col % 4
    else:
        offset = group
    tl.store(SCALES + offset, rounded, group < GROUPS)


def pack_nvfp4(
    x: torch.Tensor,
    inverse_global_scale: torch.Tensor,
    *,
    clip: float = 1.0,
    swizzled: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack contiguous CUDA BF16/FP16 matrices, including optional group clipping."""
    if x.ndim != 2 or x.shape[1] % 64 or not x.is_contiguous():
        raise ValueError("NVFP4 input must be contiguous 2-D with K divisible by 64")
    if not x.is_cuda or x.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("NVFP4 Triton packing requires CUDA BF16/FP16")
    if (
        inverse_global_scale.numel() != 1
        or inverse_global_scale.dtype != torch.float32
        or inverse_global_scale.device != x.device
    ):
        raise ValueError("Global inverse scale must be one FP32 value on input device")
    if not math.isfinite(clip) or not 0.5 <= clip <= 1.0:
        raise ValueError("Group clipping must lie in [0.5, 1]")
    rows, width = x.shape
    groups = width // 16
    packed = torch.empty((rows, width // 2), dtype=torch.uint8, device=x.device)
    shape = (triton.cdiv(rows, 128) * 128, triton.cdiv(groups, 4) * 4)
    scales = (
        torch.zeros(shape, dtype=torch.float8_e4m3fn, device=x.device)
        if swizzled
        else torch.empty((rows, groups), dtype=torch.float8_e4m3fn, device=x.device)
    )
    _pack[(triton.cdiv(rows * groups, 128),)](
        x,
        inverse_global_scale,
        packed,
        scales,
        groups,
        rows * groups,
        shape[1],
        clip,
        swizzled,
        128,
        num_warps=4,
    )
    return packed, scales
