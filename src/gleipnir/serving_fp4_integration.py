"""Explicit packed-activation integration for the pinned single-GPU dense MLP."""

from collections.abc import Callable
from types import MethodType

import torch

from gleipnir import vllm_frost_fp4 as frost
from gleipnir.cudnn_fp4_epilogue import Nvfp4ScaledGemm
from gleipnir.cudnn_fp4_gemm import PackedNvfp4
from gleipnir.serving_fp4_fusion import norm_pack, silu_pack
from gleipnir.serving_fp4_prepare import vendor_pack

_AUDIT: Callable | None = None
_WARPS = {"silu": 16, "norm": 8}


@torch.library.custom_op("gleipnir::packed_frost_linear", mutates_args=())
def packed_linear(
    codes: torch.Tensor,
    sf: torch.Tensor,
    inverse: torch.Tensor,
    weight: torch.Tensor,
    weight_sf: torch.Tensor,
    weight_inverse: torch.Tensor,
) -> torch.Tensor:
    k = codes.shape[1] * 2
    n = weight.shape[0]
    key = frost.plan_key(codes.device.index, k, n)
    if key not in frost._PLANS:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("warm packed FROST plan before capture")
        frost._PLANS[key] = Nvfp4ScaledGemm(frost.PLAN_ROWS, k, n, runtime_m=True)
    return frost._PLANS[key](
        PackedNvfp4(codes.view(torch.float4_e2m1fn_x2), sf, inverse),
        PackedNvfp4(weight, weight_sf, weight_inverse),
    )


@packed_linear.register_fake
def _fake_linear(codes, sf, inverse, weight, weight_sf, weight_inverse):
    return codes.new_empty((codes.shape[0], weight.shape[0]), dtype=torch.bfloat16)


@torch.library.custom_op("gleipnir::silu_frost_pack", mutates_args=())
def silu_producer(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a = silu_pack(x, warps=_WARPS["silu"])
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


@torch.library.custom_op("gleipnir::norm_frost_pack", mutates_args=())
def norm_producer(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    a, summed = norm_pack(x, residual, weight, eps, warps=_WARPS["norm"])
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
    if self._fp4_norm:
        if not isinstance(x, PackedNvfp4):
            raise ValueError(
                "normalization producer did not provide packed activations"
            )
        gate_up = _project(self.gate_up_proj, x)
    else:
        gate_up, _ = self.gate_up_proj(x)
    if self._fp4_silu:
        codes, sf, inverse = silu_producer(gate_up)
        out = _project(self.down_proj, PackedNvfp4(codes, sf, inverse))
    else:
        out, _ = self.down_proj(self.act_fn(gate_up))
    return out


def _norm_forward(self, x, residual=None):
    if residual is None:
        raise ValueError("post-attention packing requires the recorded residual")
    codes, sf, inverse, summed = norm_producer(
        x, residual, self.weight, self.variance_epsilon
    )
    return PackedNvfp4(codes, sf, inverse), summed


def install(model: torch.nn.Module, mode: str, *, warps: dict, audit: Callable) -> dict:
    """Install only after baseline weight loading/native checks, before compilation."""
    global _AUDIT
    if mode not in {"vendor", "silu", "norm", "combined"}:
        raise ValueError("unsupported FP4 preparation mode")
    _AUDIT = audit
    _WARPS.update(warps)
    if mode in {"vendor", "combined"}:
        original = frost.pack_operand

        def prepare(x, **kwargs):
            if kwargs != {
                "row_amax": True,
                "chunked_rows": True,
                "hardware_packing": True,
            }:
                raise ValueError("unexpected runtime FROST packing flags")
            a = vendor_pack(x)
            audit("vendor", tuple(x.shape))
            return a

        frost.pack_operand = prepare
    else:
        original = frost.pack_operand
    mlps, norms = 0, 0
    if mode in {"silu", "norm", "combined"}:
        for name, layer in model.named_modules():
            if not name.endswith(".post_attention_layernorm"):
                continue
            parent = model.get_submodule(name.rsplit(".", 1)[0])
            mlp = parent.mlp
            if (
                type(mlp).__name__ != "Qwen2MoeMLP"
                or mlp.expert_gate is not None
                or mlp.gate_up_proj.weight.shape != (18432, 1280)
                or mlp.down_proj.weight.shape != (2560, 4608)
                or type(layer).__name__ != "GemmaRMSNorm"
            ):
                raise ValueError("unsupported model scope for FP4 producer fusion")
            mlp._fp4_norm = mode in {"norm", "combined"}
            mlp._fp4_silu = mode in {"silu", "combined"}
            mlp.forward = MethodType(_mlp_forward, mlp)
            mlps += 1
            if mlp._fp4_norm:
                layer.forward = MethodType(_norm_forward, layer)
                norms += 1
        if mlps != 32 or norms != (32 if mode in {"norm", "combined"} else 0):
            raise ValueError("incomplete FP4 fusion scope")
    return {
        "mode": mode,
        "mlp_count": mlps,
        "normalization_count": norms,
        "original_packer": original.__module__ + "." + original.__name__,
    }
