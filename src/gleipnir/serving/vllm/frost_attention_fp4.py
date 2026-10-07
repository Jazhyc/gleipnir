"""FP4 full-attention linears on the established FROST MLP/GDN quantizer."""

from typing import Any

import torch
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import register_quantization_config

from gleipnir.serving_attention_fp4 import EXPECTED, SHAPES, projection_identity
from gleipnir.vllm_frost_fp4 import FrostFp4LinearMethod
from gleipnir.vllm_frost_gdn_fp4 import FrostGdnFp4Config

_AUDIT = None


class AttentionFp4Method(FrostFp4LinearMethod):
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_gleipnir_frost_ready", False):
            return
        identity = projection_identity(self.prefix)
        if identity not in EXPECTED or tuple(layer.weight.shape) != tuple(
            reversed(SHAPES[identity[1]])
        ):
            raise ValueError(f"unsupported FP4 attention geometry: {self.prefix}")
        super().process_weights_after_loading(layer)

    def apply(self, layer, x, bias=None):
        result = super().apply(layer, x, bias)
        if _AUDIT is not None and not torch.cuda.is_current_stream_capturing():
            _AUDIT(
                self.prefix,
                x.numel() // x.shape[-1],
                SHAPES[projection_identity(self.prefix)[1]],
            )
        return result


@register_quantization_config("gleipnir_frost_attention_fp4")
class FrostAttentionFp4Config(FrostGdnFp4Config):
    @classmethod
    def get_name(cls) -> str:
        return "gleipnir_frost_attention_fp4"

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "FrostAttentionFp4Config":
        return cls()

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, LinearBase) and projection_identity(prefix) is not None:
            return AttentionFp4Method(prefix)
        return super().get_quant_method(layer, prefix)
