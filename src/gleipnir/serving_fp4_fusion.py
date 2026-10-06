"""Row-scaled BF16 producer fusion with the existing FROST FP4 format."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from gleipnir.cudnn_fp4_gemm import PackedNvfp4


@triton.jit
def _emit(value, Q, SF, INV, row, M, K: tl.constexpr, BLOCK: tl.constexpr):
    columns = tl.arange(0, BLOCK)
    value = tl.where(columns < K, value, 0.0)
    maximum = tl.max(tl.abs(value), 0)
    inverse = tl.where(maximum > 0, tl.div_rn(2688.0, tl.maximum(maximum, 1e-30)), 1.0)
    groups = tl.reshape(value, (BLOCK // 16, 16))
    scale = (tl.max(tl.abs(groups), 1) * inverse / 6.0).to(tl.float8e4nv)
    denom = scale.to(tl.float32)
    normalized = groups * inverse / tl.where(denom > 0, denom, 1.0)[:, None]
    low, high = tl.split(tl.reshape(normalized, (BLOCK // 16, 8, 2)))
    packed = tl.inline_asm_elementwise(
        "{ .reg .b8 p; cvt.rn.satfinite.e2m1x2.f32 p, $2, $1; "
        "mov.b32 $0, {p, p, p, p}; }",
        constraints="=r,f,f",
        args=[low, high],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    ).to(tl.uint8)
    indexes = tl.arange(0, BLOCK // 2)
    tl.store(
        Q + row * (K // 2) + indexes,
        tl.reshape(packed, (BLOCK // 2,)),
        (row < M) & (indexes < K // 2),
    )
    g = tl.arange(0, BLOCK // 16)
    offset = (
        ((row // 128 * (K // 64) + g // 4) * 32 + row % 32) * 4 + (row % 128) // 32
    ) * 4 + g % 4
    tl.store(SF + offset, scale, g < K // 16)
    tl.store(INV + row, inverse, row < M)


@triton.jit(do_not_specialize=["M"])
def _silu_pack(X, Q, SF, INV, M, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    j = tl.arange(0, BLOCK)
    gate = tl.load(X + row * (2 * K) + j, (row < M) & (j < K), other=0).to(tl.float32)
    up = tl.load(X + row * (2 * K) + K + j, (row < M) & (j < K), other=0).to(tl.float32)
    # Match the compiled producer: BF16 rounding occurs only after the multiply.
    activated = gate / (1.0 + libdevice.exp(-gate))
    value = (activated * up).to(tl.bfloat16).to(tl.float32)
    _emit(value, Q, SF, INV, row, M, K, BLOCK)


@triton.jit(do_not_specialize=["M"])
def _norm_pack(
    X, R, W, Q, SF, INV, RES, M, EPS: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    j = tl.arange(0, BLOCK)
    x = tl.load(X + row * K + j, (row < M) & (j < K), other=0).to(tl.float32)
    r = tl.load(R + row * K + j, (row < M) & (j < K), other=0).to(tl.float32)
    summed = x + r
    tl.store(RES + row * K + j, summed.to(tl.bfloat16), (row < M) & (j < K))
    variance = tl.sum(summed * summed, 0) / K
    weight = tl.load(W + j, j < K, other=0).to(tl.float32) + 1.0
    value = (summed * tl.rsqrt(variance + EPS) * weight).to(tl.bfloat16).to(tl.float32)
    _emit(value, Q, SF, INV, row, M, K, BLOCK)


def _buffers(x: torch.Tensor, k: int):
    m = x.shape[0]
    if x.dtype != torch.bfloat16 or not x.is_cuda or not x.is_contiguous() or m < 1:
        raise ValueError("fused FP4 producer requires contiguous positive BF16 rows")
    codes = torch.empty((m, k // 2), dtype=torch.uint8, device=x.device)
    sf = torch.empty(
        (triton.cdiv(m, 128) * 128, k // 16), dtype=torch.float8_e4m3fn, device=x.device
    )
    inv = torch.empty(m, dtype=torch.float32, device=x.device)
    return codes, sf, inv


def silu_pack(x: torch.Tensor, *, warps: int = 16) -> PackedNvfp4:
    """Produce packed MLP-down activations directly from BF16 gate/up values."""
    if x.ndim != 2 or x.shape[1] != 18432 or warps not in {8, 16}:
        raise ValueError("fused SwiGLU requires the pinned K9216 dense MLP")
    k = x.shape[1] // 2
    q, sf, inv = _buffers(x, k)
    _silu_pack[(triton.cdiv(x.shape[0], 128) * 128,)](
        x,
        q,
        sf,
        inv,
        x.shape[0],
        k,
        triton.next_power_of_2(k),
        num_warps=warps,
        enable_fp_fusion=False,
    )
    return PackedNvfp4(q.view(torch.float4_e2m1fn_x2), sf, inv)


def norm_pack(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    warps: int = 8,
) -> tuple[PackedNvfp4, torch.Tensor]:
    """Fuse Qwen3.5 residual add, FP32 RMSNorm and row-scaled FP4 emission."""
    if (
        x.ndim != 2
        or x.shape[1] != 2560
        or residual.shape != x.shape
        or residual.dtype != x.dtype
        or not residual.is_contiguous()
        or weight.shape != (2560,)
        or weight.dtype not in {torch.bfloat16, torch.float32}
        or warps not in {4, 8}
    ):
        raise ValueError("fused normalization requires pinned BF16 Qwen3.5 rows")
    k = x.shape[1]
    q, sf, inv = _buffers(x, k)
    summed = torch.empty_like(x)
    _norm_pack[(triton.cdiv(x.shape[0], 128) * 128,)](
        x,
        residual,
        weight,
        q,
        sf,
        inv,
        summed,
        x.shape[0],
        eps,
        k,
        triton.next_power_of_2(k),
        num_warps=warps,
        enable_fp_fusion=False,
    )
    return PackedNvfp4(q.view(torch.float4_e2m1fn_x2), sf, inv), summed
