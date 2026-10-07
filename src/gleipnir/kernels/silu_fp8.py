"""Experimental fused BF16 SiLU/multiply and dynamic per-token E4M3 packing."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def validate_input(x: torch.Tensor) -> None:
    """Reject unsupported shapes and precision before launching a GPU kernel."""
    if x.ndim != 2 or x.shape[1] == 0 or x.shape[1] % 2:
        raise ValueError("Expected a nonempty 2-D gate/up matrix with even width")
    if x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("Expected contiguous BF16 gate/up inputs")
    if not x.is_cuda:
        raise ValueError("Fused SiLU FP8 requires CUDA")


@triton.jit
def _silu_pack(X, Q, S, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    gate = tl.load(X + row * (2 * WIDTH) + col, col < WIDTH, other=0).to(tl.float32)
    up = tl.load(X + row * (2 * WIDTH) + WIDTH + col, col < WIDTH, other=0).to(
        tl.float32
    )
    # Preserve the intermediate BF16 rounding in the unfused serving path.
    value = (gate * tl.sigmoid(gate) * up).to(tl.bfloat16).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(value), 0) / 448.0, 1.0 / (448.0 * 512.0))
    quantized = tl.minimum(tl.maximum(value / scale, -448.0), 448.0)
    tl.store(Q + row * WIDTH + col, quantized.to(tl.float8e4nv), col < WIDTH)
    tl.store(S + row, scale)


@torch.library.custom_op("gleipnir::silu_fp8", mutates_args=())
def silu_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return E4M3 activations and FP32 row scales in one Triton launch."""
    validate_input(x)
    rows, doubled_width = x.shape
    width = doubled_width // 2
    q = torch.empty((rows, width), dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.empty((rows, 1), dtype=torch.float32, device=x.device)
    if rows:
        _silu_pack[(rows,)](
            x, q, scales, width, triton.next_power_of_2(width), num_warps=8
        )
    return q, scales


@silu_fp8.register_fake
def _fake(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    validate_input(x)
    return (
        x.new_empty((x.shape[0], x.shape[1] // 2), dtype=torch.float8_e4m3fn),
        x.new_empty((x.shape[0], 1), dtype=torch.float32),
    )
