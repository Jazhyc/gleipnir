"""Fixed-order Hessian feedback for native group-16 NVFP4 weights."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _feedback_block(
    W,
    U,
    ERRORS,
    CODES,
    SCALES,
    GLOBAL,
    N: tl.constexpr,
    K: tl.constexpr,
    START,
    WIDTH: tl.constexpr,
    ROWS: tl.constexpr,
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    col = tl.arange(0, WIDTH)
    values = tl.load(
        W + row[:, None] * K + START + col[None, :], row[:, None] < N, other=0
    )
    quantized = tl.full((ROWS, WIDTH), 0, tl.int32)
    errors = tl.full((ROWS, WIDTH), 0, tl.float32)
    global_scale = tl.load(GLOBAL)
    group_scale = tl.full((ROWS,), 1.0, tl.float32)
    for index in range(WIDTH):
        if index % 16 == 0:
            maximum = tl.max(
                tl.where(
                    (col[None, :] >= index) & (col[None, :] < index + 16),
                    tl.abs(values),
                    0.0,
                ),
                1,
            )
            group_scale = (
                tl.minimum(maximum / (6.0 * global_scale), 448.0)
                .to(tl.float8e4nv)
                .to(tl.float32)
            )
            tl.store(
                SCALES + row * (K // 16) + (START + index) // 16, group_scale, row < N
            )
        current = tl.sum(tl.where(col[None, :] == index, values, 0.0), 1)
        normalized = tl.abs(current) / tl.where(
            group_scale > 0, group_scale * global_scale, 1.0
        )
        code = (
            (normalized > 0.25).to(tl.int32)
            + (normalized >= 0.75).to(tl.int32)
            + (normalized > 1.25).to(tl.int32)
            + (normalized >= 1.75).to(tl.int32)
            + (normalized > 2.5).to(tl.int32)
            + (normalized >= 3.5).to(tl.int32)
            + (normalized > 5.0).to(tl.int32)
        )
        decoded = tl.where(
            code == 7,
            6.0,
            tl.where(
                code == 6,
                4.0,
                tl.where(code == 5, 3.0, tl.where(code == 4, 2.0, code * 0.5)),
            ),
        )
        reconstructed = (
            decoded * group_scale * global_scale * tl.where(current < 0, -1.0, 1.0)
        )
        signs = (current.to(tl.int32, bitcast=True) >> 28) & 8
        quantized = tl.where(
            col[None, :] == index, code[:, None] | signs[:, None], quantized
        )
        diagonal = tl.load(U + index * WIDTH + index)
        error = (current - reconstructed) / diagonal
        errors = tl.where(col[None, :] == index, error[:, None], errors)
        factor = tl.load(U + index * WIDTH + col)
        values -= error[:, None] * tl.where(col >= index, factor, 0.0)[None, :]
    tl.store(ERRORS + row[:, None] * WIDTH + col[None, :], errors, row[:, None] < N)
    low, high = tl.split(quantized.reshape((ROWS, WIDTH // 2, 2)))
    tl.store(
        CODES
        + row[:, None] * (K // 2)
        + START // 2
        + tl.arange(0, WIDTH // 2)[None, :],
        (low | (high << 4)).to(tl.uint8),
        row[:, None] < N,
    )


def hessian_factor(
    x: torch.Tensor, damping: float = 0.01
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the upper Cholesky factor of the damped inverse input covariance."""
    if x.ndim != 2 or not 0 < damping <= 1 or not torch.isfinite(x).all().item():
        raise ValueError("Invalid GPTQ calibration matrix/damping")
    hessian = x.float().t() @ x.float() / x.shape[0]
    diagonal = torch.diagonal(hessian)
    dead = diagonal == 0
    diagonal[dead] = 1
    diagonal.add_(damping * diagonal.mean())
    inverse = torch.cholesky_inverse(torch.linalg.cholesky(hessian))
    factor = torch.linalg.cholesky(inverse, upper=True).contiguous()
    if not torch.isfinite(factor).all().item():
        raise ValueError("Nonfinite Hessian factor")
    return factor, dead


def quantize_gptq(
    weight: torch.Tensor, x: torch.Tensor, *, damping: float = 0.01, block: int = 128
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Quantize unchanged weights in fixed K order using disjoint calibration inputs."""
    if not weight.is_cuda or weight.ndim != 2 or weight.shape[1] % block or block % 16:
        raise ValueError(
            "GPTQ requires CUDA matrices and complete group-aligned blocks"
        )
    if x.device != weight.device or x.shape[1] != weight.shape[1]:
        raise ValueError("GPTQ calibration shape/device mismatch")
    if not torch.isfinite(weight).all().item() or weight.dtype not in (
        torch.float16,
        torch.bfloat16,
    ):
        raise ValueError("GPTQ source must be finite BF16/FP16")
    factor, dead = hessian_factor(x, damping)
    rows, width = weight.shape
    working = weight.float().clone()
    working[:, dead] = 0
    scale = weight.abs().amax().float().clamp_min(1e-12).reshape(1) / (6 * 448)
    packed = torch.empty((rows, width // 2), device=weight.device, dtype=torch.uint8)
    scales = torch.empty(
        (rows, width // 16), device=weight.device, dtype=torch.float8_e4m3fn
    )
    errors = torch.empty((rows, block), device=weight.device)
    for start in range(0, width, block):
        upper = factor[start : start + block, start : start + block].contiguous()
        _feedback_block[(triton.cdiv(rows, 4),)](
            working,
            upper,
            errors,
            packed,
            scales,
            scale,
            rows,
            width,
            start,
            block,
            4,
            num_warps=4,
        )
        if start + block < width:
            working[:, start + block :] -= (
                errors @ factor[start : start + block, start + block :]
            )
    torch.cuda.synchronize()
    if not torch.isfinite(scales.float()).all().item():
        raise ValueError("Nonfinite GPTQ packed scale")
    return (
        packed,
        scales,
        scale,
        {
            "damping": damping,
            "block_size": block,
            "act_order": False,
            "dead_channels": int(dead.sum().item()),
            "calibration_vectors": x.shape[0],
        },
    )
