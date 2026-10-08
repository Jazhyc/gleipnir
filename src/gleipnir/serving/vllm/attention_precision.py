"""Native BF16/FP8 full-attention projections with unchanged FP4 MLP/GDN."""

from __future__ import annotations

from collections.abc import Callable

import torch
from vllm import _custom_ops as ops
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.online.fp8 import Fp8PtpcOnlineLinearMethod

from gleipnir.serving.fp4.attention import EXPECTED, SHAPES, projection_identity
from gleipnir.serving.vllm.frost_attention_fp4 import FrostAttentionFp4Config

_NAMES: dict[int, str] = {}
_AUDIT: Callable[[str, int, str], None] | None = None


def install_audit(callback: Callable[[str, int, str], None]) -> None:
    """Observe actual native calls outside compilation and graph capture."""
    global _AUDIT
    _AUDIT = callback


def observe(x: torch.Tensor, weight: torch.Tensor, precision: str) -> None:
    if _AUDIT is not None and not torch.cuda.is_current_stream_capturing():
        name = _NAMES.get(weight.data_ptr())
        if name is not None:
            _AUDIT(name, x.numel() // x.shape[-1], precision)


@torch.library.custom_op("gleipnir::attention_bf16_linear", mutates_args=())
def bf16_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    result = torch.nn.functional.linear(x, weight)
    observe(x, weight, "bf16")
    return result


@bf16_linear.register_fake
def fake_bf16(x, weight):
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


@torch.library.custom_op("gleipnir::attention_fp8_linear", mutates_args=())
def fp8_linear(
    x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    matrix = x.reshape(-1, x.shape[-1]).contiguous()
    codes, activation_scale = ops.scaled_fp8_quant(
        matrix, scale=None, use_per_token_if_dynamic=True
    )
    result = ops.cutlass_scaled_mm(
        codes,
        weight,
        scale_a=activation_scale,
        scale_b=scale,
        out_dtype=torch.bfloat16,
    )
    observe(x, weight, "fp8")
    return result.reshape(*x.shape[:-1], weight.shape[1])


@fp8_linear.register_fake
def fake_fp8(x, weight, scale):
    return x.new_empty((*x.shape[:-1], weight.shape[1]), dtype=torch.bfloat16)


def check_geometry(prefix: str, shape: tuple[int, ...]) -> None:
    identity = projection_identity(prefix)
    if identity not in EXPECTED or shape != tuple(reversed(SHAPES[identity[1]])):
        raise ValueError(f"unsupported full-attention projection: {prefix}, {shape}")


class AttentionBf16Method(UnquantizedLinearMethod):
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        check_geometry(self.prefix, tuple(layer.weight.shape))
        if layer.weight.dtype != torch.bfloat16 or layer.bias is not None:
            raise ValueError("BF16 attention requires bias-free BF16 source weights")
        _NAMES[layer.weight.data_ptr()] = self.prefix

    def apply(self, layer, x, bias=None):
        if bias is not None:
            raise ValueError("BF16 attention projection bias is unsupported")
        return bf16_linear(x, layer.weight)


class AttentionFp8Method(Fp8PtpcOnlineLinearMethod):
    def __init__(self, prefix: str) -> None:
        super().__init__()
        self.prefix = prefix

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        check_geometry(self.prefix, tuple(layer.weight.shape))
        original = layer.weight
        if original.dtype != torch.bfloat16 or layer.bias is not None:
            raise ValueError("FP8 attention must quantize original merged BF16 weights")
        source = original[:16].float().clone()
        super().process_weights_after_loading(layer)
        if (
            type(self.fp8_linear).__name__ != "CutlassFP8ScaledMMLinearKernel"
            or layer.weight.dtype != torch.float8_e4m3fn
            or not bool(torch.isfinite(layer.weight_scale).all())
            or not bool((layer.weight_scale > 0).all())
        ):
            raise ValueError("FP8 attention requires native per-token/channel CUTLASS")
        decoded = layer.weight[:, :16].float().t() * layer.weight_scale[:16]
        layer._attention_fp8_weight_error = float(
            (decoded - source).norm() / source.norm().clamp_min(1e-12)
        )
        _NAMES[layer.weight.data_ptr()] = self.prefix

    def apply(self, layer, x, bias=None):
        if bias is not None:
            raise ValueError("FP8 attention projection bias is unsupported")
        return fp8_linear(x, layer.weight, layer.weight_scale)


@register_quantization_config("gleipnir_frost_attention_precision")
class AttentionPrecisionConfig(FrostAttentionFp4Config):
    @classmethod
    def get_name(cls) -> str:
        return "gleipnir_frost_attention_precision"

    @classmethod
    def from_config(cls, config: dict) -> AttentionPrecisionConfig:
        return cls()

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, LinearBase) and projection_identity(prefix) is not None:
            precision = get_current_vllm_config().additional_config[
                "serving_condition"
            ]["attention_projection_precision"]
            if precision == "bf16":
                return AttentionBf16Method(prefix)
            if precision == "fp8":
                return AttentionFp8Method(prefix)
            if precision != "fp4":
                raise ValueError("unknown attention projection precision")
        return super().get_quant_method(layer, prefix)
