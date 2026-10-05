"""Experimental frozen-base NVFP4 GEMM primitives.

Native operations have no registered autograd. Training callers must supply an
explicit autograd wrapper while retaining separate FP32 adapter masters.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from gleipnir.nvfp4_pack import pack_nvfp4


@triton.jit
def _weight_tiles(X, INV, Q, SF, ROWS: tl.constexpr, K: tl.constexpr):
    tile = tl.program_id(0)
    r = tile // (K // 16) * 16 + tl.arange(0, 16)
    c = tile % (K // 16) * 16 + tl.arange(0, 16)
    v = tl.load(X + r[:, None] * K + c[None, :]).to(tl.float32)
    inv = tl.load(INV)
    scale = (
        (tl.max(tl.max(tl.abs(v), 0), 0) * inv / 6.0).to(tl.float8e4nv).to(tl.float32)
    )
    a = tl.abs(v) * inv / tl.where(scale > 0, scale, 1.0)
    code = (
        (a > 0.25).to(tl.int32)
        + (a >= 0.75).to(tl.int32)
        + (a > 1.25).to(tl.int32)
        + (a >= 1.75).to(tl.int32)
        + (a > 2.5).to(tl.int32)
        + (a >= 3.5).to(tl.int32)
        + (a > 5.0).to(tl.int32)
    )
    code = (code | ((v.to(tl.int32, bitcast=True) >> 28) & 8)).to(tl.uint8)
    lo, hi = tl.split(tl.reshape(code, (16, 8, 2)))
    tl.store(
        Q + r[:, None] * (K // 2) + (tile % (K // 16) * 8 + tl.arange(0, 8))[None, :],
        lo | (hi << 4),
    )
    col = tile % (K // 16)
    offset = (
        ((r // 128 * (K // 64) + col // 4) * 32 + r % 32) * 4 + (r % 128) // 32
    ) * 4 + col % 4
    tl.store(SF + offset, scale)


@triton.jit
def _amax_partials(X, P, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(X + p, p < SIZE, other=0).to(tl.float32)
    tl.store(P + tl.program_id(0), tl.max(tl.abs(v), 0))


@triton.jit
def _inverse_amax(P, INV, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    p = tl.arange(0, BLOCK)
    amax = tl.max(tl.load(P + p, p < SIZE, other=0), 0)
    tl.store(INV, tl.where(amax > 0, tl.div_rn(2688.0, tl.maximum(amax, 1e-30)), 1.0))


@triton.jit
def _pack_rows(X, Q, SF, INV, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    v = tl.load(X + row * K + k, k < K, other=0).to(tl.float32)
    amax = tl.max(tl.abs(v), 0)
    inv = tl.where(amax > 0, tl.div_rn(2688.0, tl.maximum(amax, 1e-30)), 1.0)
    tl.store(INV + row, inv)
    local = tl.max(tl.reshape(tl.abs(v), (BLOCK // 16, 16)), 1)
    scales = (local * inv / 6.0).to(tl.float8e4nv).to(tl.float32)
    scale = tl.reshape(tl.broadcast_to(scales[:, None], (BLOCK // 16, 16)), (BLOCK,))
    a = tl.abs(v) * inv / tl.where(scale > 0, scale, 1.0)
    code = (
        (a > 0.25).to(tl.int32)
        + (a >= 0.75).to(tl.int32)
        + (a > 1.25).to(tl.int32)
        + (a >= 1.75).to(tl.int32)
        + (a > 2.5).to(tl.int32)
        + (a >= 3.5).to(tl.int32)
        + (a > 5.0).to(tl.int32)
    )
    code = (code | ((v.to(tl.int32, bitcast=True) >> 28) & 8)).to(tl.uint8)
    lo, hi = tl.split(tl.reshape(code, (BLOCK // 2, 2)))
    j = tl.arange(0, BLOCK // 2)
    tl.store(Q + row * (K // 2) + j, lo | (hi << 4), j < K // 2)
    c = tl.arange(0, BLOCK // 16)
    offset = (
        ((row // 128 * (K // 64) + c // 4) * 32 + row % 32) * 4 + (row % 128) // 32
    ) * 4 + c % 4
    tl.store(SF + offset, scales, c < K // 16)


@triton.jit
def _row_inverse(X, INV, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    j = tl.arange(0, BLOCK)
    amax = tl.max(tl.abs(tl.load(X + row * K + j, j < K, other=0).to(tl.float32)), 0)
    tl.store(
        INV + row, tl.where(amax > 0, tl.div_rn(2688.0, tl.maximum(amax, 1e-30)), 1.0)
    )


@triton.jit
def _pack_row_blocks(
    X, INV, Q, SF, K: tl.constexpr, GROUPS: tl.constexpr, BLOCK: tl.constexpr
):
    g = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    j = tl.arange(0, 16)
    v = tl.load(X + g[:, None] * 16 + j[None, :], g[:, None] < GROUPS, other=0).to(
        tl.float32
    )
    inv = tl.load(INV + g // (K // 16), g < GROUPS, other=1)
    scale = (tl.max(tl.abs(v), 1) * inv / 6.0).to(tl.float8e4nv).to(tl.float32)
    a = tl.abs(v) * inv[:, None] / tl.where(scale > 0, scale, 1.0)[:, None]
    code = (
        (a > 0.25).to(tl.int32)
        + (a >= 0.75).to(tl.int32)
        + (a > 1.25).to(tl.int32)
        + (a >= 1.75).to(tl.int32)
        + (a > 2.5).to(tl.int32)
        + (a >= 3.5).to(tl.int32)
        + (a > 5.0).to(tl.int32)
    )
    code = (code | ((v.to(tl.int32, bitcast=True) >> 28) & 8)).to(tl.uint8)
    lo, hi = tl.split(tl.reshape(code, (BLOCK, 8, 2)))
    tl.store(
        Q + g[:, None] * 8 + tl.arange(0, 8)[None, :],
        lo | (hi << 4),
        g[:, None] < GROUPS,
    )
    r, c = g // (K // 16), g % (K // 16)
    offset = (
        ((r // 128 * (K // 64) + c // 4) * 32 + r % 32) * 4 + (r % 128) // 32
    ) * 4 + c % 4
    tl.store(SF + offset, scale, g < GROUPS)


PACKING_KERNEL_METADATA: dict = {}


def _kernel_metadata(kernel) -> dict:
    return {
        "registers": getattr(kernel, "n_regs", None),
        "spills": getattr(kernel, "n_spills", None),
    }


@triton.jit
def _descale(
    X,
    Y,
    IA,
    IB,
    SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    N: tl.constexpr,
    ROW_SCALE: tl.constexpr,
):
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + p, p < SIZE, other=0).to(tl.float32)
    if ROW_SCALE:
        inv_a = tl.load(IA + p // N, p < SIZE, other=1)
    else:
        inv_a = tl.load(IA)
    scale = 1.0 / (inv_a * tl.load(IB))
    tl.store(Y + p, x * scale, p < SIZE)


@dataclass
class PackedNvfp4:
    """Typed packed codes, swizzled E4M3 block scales and global inverse scale."""

    codes: torch.Tensor
    scales: torch.Tensor
    inverse: torch.Tensor


def validate_weight_shape(shape: tuple[int, ...]) -> None:
    """Require complete 16x16 tiles and complete scale atoms along K."""
    if len(shape) != 2 or min(shape) <= 0 or shape[0] % 16 or shape[1] % 64:
        raise ValueError(
            "weight needs positive rows divisible by 16 and K divisible by 64"
        )


def pack_operand(
    x: torch.Tensor,
    *,
    weight: bool = False,
    fused_amax: bool = False,
    row_amax: bool = False,
    chunked_rows: bool = False,
) -> PackedNvfp4:
    """Pack BF16 CUDA data with dynamic global scaling and RNE E2M1 codes."""
    if x.requires_grad and torch.is_grad_enabled():
        raise ValueError("native FP4 packing requires explicit autograd integration")
    if (
        x.ndim != 2
        or not x.is_contiguous()
        or not x.is_cuda
        or x.dtype != torch.bfloat16
        or min(x.shape) <= 0
        or x.shape[1] % 64
    ):
        raise ValueError("packing requires a contiguous 2-D CUDA BF16 operand")
    if row_amax:
        if weight or x.shape[0] <= 0 or x.shape[1] % 64:
            raise ValueError("row amax needs activation rows and K divisible by 64")
        rows, k = x.shape
        q = torch.empty((rows, k // 2), dtype=torch.uint8, device=x.device)
        sf = torch.zeros(
            ((rows + 127) // 128 * 128, k // 16),
            dtype=torch.float8_e4m3fn,
            device=x.device,
        )
        inverse = torch.empty(rows, dtype=torch.float32, device=x.device)
        if chunked_rows:
            reduction = _row_inverse[(rows,)](
                x, inverse, k, triton.next_power_of_2(k), num_warps=8
            )
            packing = _pack_row_blocks[(triton.cdiv(rows * k // 16, 128),)](
                x, inverse, q, sf, k, rows * k // 16, 128, num_warps=4
            )
            PACKING_KERNEL_METADATA[(k, True)] = {
                "reduction": _kernel_metadata(reduction),
                "packing": _kernel_metadata(packing),
            }
        else:
            packing = _pack_rows[(rows,)](
                x, q, sf, inverse, k, triton.next_power_of_2(k), num_warps=8
            )
            PACKING_KERNEL_METADATA[(k, False)] = {"packing": _kernel_metadata(packing)}
        return PackedNvfp4(q.view(torch.float4_e2m1fn_x2), sf, inverse)
    if weight:
        validate_weight_shape(tuple(x.shape))
    if fused_amax:
        size = triton.cdiv(x.numel(), 16384)
        partials = torch.empty(size, dtype=torch.float32, device=x.device)
        inverse = torch.empty(1, dtype=torch.float32, device=x.device)
        _amax_partials[(size,)](x, partials, x.numel(), 16384)
        _inverse_amax[(1,)](partials, inverse, size, triton.next_power_of_2(size))
    else:
        amax = x.abs().amax().float()
        inverse = torch.where(
            amax > 0, torch.div(2688.0, amax.clamp_min(1e-30)), 1.0
        ).reshape(1)
    if weight:
        rows, k = x.shape
        q = torch.empty((rows, k // 2), dtype=torch.uint8, device=x.device)
        sf = torch.zeros(
            ((rows + 127) // 128 * 128, k // 16),
            dtype=torch.float8_e4m3fn,
            device=x.device,
        )
        _weight_tiles[(rows // 16 * (k // 16),)](
            x, inverse, q, sf, rows, k, num_warps=4
        )
    else:
        q, sf = pack_nvfp4(x, inverse, swizzled=True)
    return PackedNvfp4(q.view(torch.float4_e2m1fn_x2), sf, inverse)


def decode_operand(p: PackedNvfp4) -> torch.Tensor:
    """Slow independent diagnostic decoder, excluded from measured native paths."""
    q = p.codes.view(torch.uint8)
    rows, half_k = q.shape
    k = half_k * 2
    r = torch.arange(rows, device=q.device)[:, None]
    c = torch.arange(k // 16, device=q.device)[None, :]
    offsets = (
        ((r // 128 * (k // 64) + c // 4) * 32 + r % 32) * 4 + (r % 128) // 32
    ) * 4 + c % 4
    scales = p.scales.flatten().float()[offsets]
    values = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=q.device,
        dtype=torch.float32,
    )
    codes = torch.stack((q & 15, q >> 4), dim=-1).reshape(rows, k).long()
    return (
        values[codes] * scales.repeat_interleave(16, dim=1) / p.inverse.reshape(-1, 1)
    )


class Nvfp4Gemm:
    """Pinned cuDNN FROST NVFP4 GEMM, BF16 output and explicit caller stream."""

    def __init__(self, m: int, k: int, n: int, *, tile: str = "auto") -> None:
        import cudnn
        from cudnn.gated_attention_block.kernels.proj_gemm import build_proj_gemm

        self.plan = build_proj_gemm(
            m=m,
            k=k,
            n=n,
            dtype=torch.float4_e2m1fn_x2,
            label=f"gleipnir_nvfp4_{m}_{k}_{n}",
            block_scale=True,
            block_size=16,
            sf_dtype=cudnn.data_type.FP8_E4M3,
            tile_config=tile,
            out_dtype=torch.bfloat16,
        )
        self.workspace = torch.empty(
            self.plan.workspace_bytes, device="cuda", dtype=torch.uint8
        )

    def __call__(self, a: PackedNvfp4, b: PackedNvfp4) -> torch.Tensor:
        from cudnn.gated_attention_block.kernels.proj_gemm import run_proj_gemm

        if b.inverse.numel() != 1 or a.inverse.numel() not in {1, self.plan.m}:
            raise ValueError(
                "GEMM needs scalar weight scale and scalar/per-row activation scales"
            )
        for operand in (a, b):
            if (
                operand.inverse.device != operand.codes.device
                or operand.inverse.dtype != torch.float32
            ):
                raise ValueError(
                    "global inverse scales must be FP32 on the operand device"
                )
        out = torch.empty(
            (self.plan.m, self.plan.n), device=a.codes.device, dtype=torch.bfloat16
        )
        run_proj_gemm(
            self.plan,
            a.codes,
            b.codes,
            out,
            self.workspace,
            sf_a=a.scales,
            sf_w=b.scales,
            stream=torch.cuda.current_stream(out.device).cuda_stream,
        )
        result = torch.empty_like(out)
        _descale[(triton.cdiv(out.numel(), 1024),)](
            out,
            result,
            a.inverse,
            b.inverse,
            out.numel(),
            1024,
            self.plan.n,
            a.inverse.numel() != 1,
        )
        return result
