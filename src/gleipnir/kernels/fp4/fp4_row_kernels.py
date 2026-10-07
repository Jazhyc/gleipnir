"""Fuse row scaling while retaining the stable BF16 rounding boundaries."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _normalize_rows(X, Y, S, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    values = tl.load(X + row * N + columns, columns < N, other=0).to(tl.float32)
    scale = tl.max(tl.abs(values), axis=0)
    scale = tl.where(scale == 0, 1.0, scale)
    # div_rn preserves PyTorch's FP32 division before the BF16 rounding.
    normalized = tl.div_rn(values, scale).to(tl.bfloat16)
    tl.store(Y + row * N + columns, normalized, columns < N)
    tl.store(S + row, scale)


@triton.jit
def _rescale_rows(X, S, Y, N: tl.constexpr, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(X + offsets, offsets < TOTAL, other=0).to(tl.float32)
    scales = tl.load(S + offsets // N, offsets < TOTAL, other=1)
    tl.store(Y + offsets, (values * scales).to(tl.bfloat16), offsets < TOTAL)


def normalize_rows(inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return BF16 normalized inputs and exact FP32 per-row absolute maxima."""
    if inputs.ndim != 2 or inputs.dtype != torch.bfloat16 or not inputs.is_cuda:
        raise ValueError("fused row normalization requires a CUDA BF16 matrix")
    if not inputs.is_contiguous() or not 0 < inputs.shape[1] <= 32768:
        raise ValueError("fused row normalization requires contiguous supported rows")
    rows, columns = inputs.shape
    output = torch.empty_like(inputs)
    scales = torch.empty((rows, 1), device=inputs.device, dtype=torch.float32)
    _normalize_rows[(rows,)](
        inputs,
        output,
        scales,
        columns,
        triton.next_power_of_2(columns),
        num_warps=8 if columns > 4096 else 4,
        enable_fp_fusion=False,
    )
    return output, scales


def rescale_rows(inputs: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Multiply the BF16 GEMM result by FP32 row scales and round once to BF16."""
    if inputs.ndim != 2 or inputs.dtype != torch.bfloat16 or not inputs.is_cuda:
        raise ValueError("fused row rescaling requires a CUDA BF16 matrix")
    if (
        not inputs.is_contiguous()
        or scales.shape != (inputs.shape[0], 1)
        or scales.dtype != torch.float32
        or scales.device != inputs.device
        or not scales.is_contiguous()
    ):
        raise ValueError("fused row rescaling requires contiguous matching FP32 scales")
    output = torch.empty_like(inputs)
    _rescale_rows[(triton.cdiv(inputs.numel(), 1024),)](
        inputs,
        scales,
        output,
        inputs.shape[1],
        inputs.numel(),
        1024,
        enable_fp_fusion=False,
    )
    return output
