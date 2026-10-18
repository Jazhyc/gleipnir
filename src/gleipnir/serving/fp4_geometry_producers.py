"""Config-derived fused FP4 activation producers for Qwen3.5-family MLPs.

The 4B producers (``gleipnir.serving.fp4.fusion``/``prepare``/``integration``)
are hash-bound and pin K9216 SwiGLU, K2560 RMSNorm, three vendor K values and
32 4B-shaped MLPs. Their Triton kernels take K as a compile-time constant and
are reused by import, as is the shape-free ``packed_frost_linear`` op. This
module adds K-checked wrappers, separately named custom ops and an
``install`` whose expected MLP/norm shapes and counts come from the config.
"""

from __future__ import annotations

from collections.abc import Callable
from types import MethodType
from typing import Any

import torch
import triton

from gleipnir.kernels.fp4.cudnn_fp4_gemm import PackedNvfp4
from gleipnir.serving.fp4.fusion import _buffers, _norm_pack, _silu_pack
from gleipnir.serving.fp4.integration import packed_linear
from gleipnir.serving.fp4.prepare import _vendor_finish
from gleipnir.serving.vllm import frost_fp4 as frost

SILU_WARPS = (8, 16, 32)
NORM_WARPS = (4, 8, 16)
_AUDIT: Callable | None = None
_STATE = {"silu_warps": 16, "norm_warps": 8, "silu_k": 0, "norm_k": 0}


def producer_k(text: Any) -> dict[str, int]:
    """Activation widths: SwiGLU output (MLP down input) and RMSNorm rows."""
    return {"silu": text.intermediate_size, "norm": text.hidden_size}


def vendor_k(text: Any) -> set[int]:
    """Activation widths the vendor packer serves: the large GDN projections."""
    value = text.linear_num_value_heads * text.linear_value_head_dim
    return {text.hidden_size, value}


def _check_k(k: int) -> None:
    # The 128x4 scale swizzle addresses K // 64 blocks of four 16-value groups.
    if k < 64 or k % 64:
        raise ValueError(f"FP4 producer K must be a positive multiple of 64: {k}")


def silu_pack(x: torch.Tensor, k: int, *, warps: int) -> PackedNvfp4:
    """Packed MLP-down activations directly from BF16 [gate | up] rows."""
    _check_k(k)
    if x.ndim != 2 or x.shape[1] != 2 * k or warps not in SILU_WARPS:
        raise ValueError(f"fused SwiGLU requires [gate | up] rows of width {2 * k}")
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
    k: int,
    *,
    warps: int,
) -> tuple[PackedNvfp4, torch.Tensor]:
    """Residual add, FP32 Gemma RMSNorm and row-scaled FP4 emission."""
    _check_k(k)
    if (
        x.ndim != 2
        or x.shape[1] != k
        or residual.shape != x.shape
        or residual.dtype != x.dtype
        or not residual.is_contiguous()
        or weight.shape != (k,)
        or weight.dtype not in {torch.bfloat16, torch.float32}
        or warps not in NORM_WARPS
    ):
        raise ValueError(f"fused normalization requires BF16 rows of width {k}")
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


def vendor_pack(x: torch.Tensor, allowed: set[int]) -> PackedNvfp4:
    """FlashInfer CUDA per-token NVFP4 with the 4B zero-row/inverse repair."""
    from flashinfer.quantization import SfLayout, nvfp4_quantize

    if (
        x.ndim != 2
        or x.dtype != torch.bfloat16
        or not x.is_cuda
        or not x.is_contiguous()
        or x.shape[1] not in allowed
        or x.shape[0] < 1
    ):
        raise ValueError("vendor FROST packing requires supported BF16 activation rows")
    m, k = x.shape
    _check_k(k)
    codes, scales, token_scale = nvfp4_quantize(
        x,
        1.0 / 2688.0,
        sfLayout=SfLayout.layout_128x4,
        backend="cuda",
        per_token_activation=True,
        enable_pdl=False,
    )
    scales = scales.view(torch.float8_e4m3fn).reshape(
        triton.cdiv(m, 128) * 128, k // 16
    )
    inverse = torch.empty(m, device=x.device, dtype=torch.float32)
    _vendor_finish[(triton.cdiv(m, 128) * 8,)](
        token_scale, inverse, scales, codes, m, k, 16, num_warps=4
    )
    return PackedNvfp4(codes.view(torch.float4_e2m1fn_x2), scales, inverse)


@torch.library.custom_op("gleipnir::geometry_silu_pack", mutates_args=())
def silu_producer(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a = silu_pack(x, _STATE["silu_k"], warps=_STATE["silu_warps"])
    if _AUDIT is not None:
        _AUDIT("silu", tuple(x.shape))
    return a.codes.view(torch.uint8), a.scales, a.inverse


@silu_producer.register_fake
def _fake_silu(x):
    m, k = x.shape[0], x.shape[1] // 2
    return (
        x.new_empty((m, k // 2), dtype=torch.uint8),
        x.new_empty((((m + 127) // 128) * 128, k // 16), dtype=torch.float8_e4m3fn),
        x.new_empty((m,), dtype=torch.float32),
    )


@torch.library.custom_op("gleipnir::geometry_norm_pack", mutates_args=())
def norm_producer(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    a, summed = norm_pack(
        x, residual, weight, eps, _STATE["norm_k"], warps=_STATE["norm_warps"]
    )
    if _AUDIT is not None:
        _AUDIT("norm", tuple(x.shape))
    return a.codes.view(torch.uint8), a.scales, a.inverse, summed


@norm_producer.register_fake
def _fake_norm(x, residual, weight, eps):
    m, k = x.shape
    return (
        x.new_empty((m, k // 2), dtype=torch.uint8),
        x.new_empty((((m + 127) // 128) * 128, k // 16), dtype=torch.float8_e4m3fn),
        x.new_empty((m,), dtype=torch.float32),
        x.new_empty(x.shape),
    )


def _project(layer, activation):
    if layer.bias is not None:
        raise ValueError("packed FROST adapter requires bias-free dense projections")
    return packed_linear(
        activation.codes,
        activation.scales,
        activation.inverse,
        layer.weight,
        layer.weight_scale,
        layer.weight_inverse,
    )


def _mlp_forward(self, x):
    if not isinstance(x, PackedNvfp4):
        raise ValueError("normalization producer did not provide packed activations")
    gate_up = _project(self.gate_up_proj, x)
    codes, sf, inverse = silu_producer(gate_up)
    return _project(self.down_proj, PackedNvfp4(codes, sf, inverse))


def _norm_forward(self, x, residual=None):
    if residual is None:
        raise ValueError("post-attention packing requires the recorded residual")
    codes, sf, inverse, summed = norm_producer(
        x, residual, self.weight, self.variance_epsilon
    )
    return PackedNvfp4(codes, sf, inverse), summed


def install(
    model: torch.nn.Module, text: Any, *, warps: dict[str, int], audit: Callable
) -> dict:
    """Combined mode: fused norm and SwiGLU producers, vendor GDN packing.

    Install after weight loading and native checks, before compilation.
    """
    global _AUDIT
    widths = producer_k(text)
    allowed = vendor_k(text)
    hidden, mlp_width = widths["norm"], widths["silu"]
    for k in (*widths.values(), *allowed):
        _check_k(k)
    _AUDIT = audit
    _STATE.update(
        silu_warps=warps["silu"],
        norm_warps=warps["norm"],
        silu_k=mlp_width,
        norm_k=hidden,
    )
    original = frost.pack_operand

    def prepare(x, **kwargs):
        if kwargs != {"row_amax": True, "chunked_rows": True, "hardware_packing": True}:
            raise ValueError("unexpected runtime FROST packing flags")
        a = vendor_pack(x, allowed)
        audit("vendor", tuple(x.shape))
        return a

    frost.pack_operand = prepare
    layers = len(text.layer_types)
    mlps = 0
    for name, layer in model.named_modules():
        if not name.endswith(".post_attention_layernorm"):
            continue
        mlp = model.get_submodule(name.rsplit(".", 1)[0]).mlp
        # Packed FP4 weights store two codes per byte along K.
        if (
            type(mlp).__name__ != "Qwen2MoeMLP"
            or mlp.expert_gate is not None
            or tuple(mlp.gate_up_proj.weight.shape) != (2 * mlp_width, hidden // 2)
            or tuple(mlp.down_proj.weight.shape) != (hidden, mlp_width // 2)
            or type(layer).__name__ != "GemmaRMSNorm"
        ):
            raise ValueError(f"unsupported model scope for FP4 producer fusion: {name}")
        mlp.forward = MethodType(_mlp_forward, mlp)
        layer.forward = MethodType(_norm_forward, layer)
        mlps += 1
    if mlps != layers:
        raise ValueError(f"incomplete FP4 fusion scope: {mlps} != {layers}")
    return {
        "mode": "combined",
        "mlp_count": mlps,
        "normalization_count": mlps,
        "vendor_k": sorted(allowed),
        "producer_k": widths,
        "warps": dict(warps),
        "original_packer": original.__module__ + "." + original.__name__,
    }
