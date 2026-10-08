"""Checked FP8 GDN with actual-call auditing and unchanged attention/MLP math."""

from collections.abc import Callable

import torch
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import register_quantization_config

from gleipnir.serving.gdn_precision import check_geometry, projection_identity
from gleipnir.serving.vllm.attention_precision import (
    AttentionPrecisionConfig,
    fp8_linear,
)
from gleipnir.serving.vllm.frost_gdn import CheckedGdnFp8Method

_NAMES: dict[int, str] = {}
_AUDIT: Callable | None = None


def install_audit(callback: Callable) -> None:
    global _AUDIT
    _AUDIT = callback


@torch.library.custom_op("gleipnir::gdn_fp8_linear", mutates_args=())
def gdn_fp8_linear(
    x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    result = fp8_linear(x, weight, scale)
    if _AUDIT is not None and not torch.cuda.is_current_stream_capturing():
        prefix = _NAMES.get(weight.data_ptr())
        if prefix is not None:
            _AUDIT(prefix, x.numel() // x.shape[-1])
    return result


@gdn_fp8_linear.register_fake
def fake_gdn(x, weight, scale):
    return x.new_empty((*x.shape[:-1], weight.shape[1]), dtype=torch.bfloat16)


class AuditedGdnFp8Method(CheckedGdnFp8Method):
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        check_geometry(self.prefix, tuple(layer.weight.shape))
        if layer.weight.dtype != torch.bfloat16 or layer.bias is not None:
            raise ValueError("GDN FP8 requires original bias-free merged BF16 weights")
        source = layer.weight[:16].float().clone()
        super().process_weights_after_loading(layer)
        decoded = layer.weight[:, :16].float().t() * layer.weight_scale[:16]
        layer._gdn_fp8_weight_error = float(
            (decoded - source).norm() / source.norm().clamp_min(1e-12)
        )
        _NAMES[layer.weight.data_ptr()] = self.prefix

    def apply(self, layer, x, bias=None):
        if bias is not None:
            raise ValueError("GDN FP8 bias is unsupported")
        return gdn_fp8_linear(x, layer.weight, layer.weight_scale)


@register_quantization_config("gleipnir_frost_gdn_precision")
class GdnPrecisionConfig(AttentionPrecisionConfig):
    @classmethod
    def get_name(cls) -> str:
        return "gleipnir_frost_gdn_precision"

    @classmethod
    def from_config(cls, config: dict):
        return cls()

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, LinearBase) and projection_identity(prefix) is not None:
            condition = get_current_vllm_config().additional_config["serving_condition"]
            if condition["gdn_projection_precision"] != "fp8":
                raise ValueError("candidate requires FP8 GDN projections")
            return AuditedGdnFp8Method(prefix)
        return super().get_quant_method(layer, prefix)
