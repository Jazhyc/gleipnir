"""Checked FROST FP4 GDN projections with BF16 gates and recurrence operands."""

import re
from typing import Any

import torch
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase

from gleipnir.cudnn_fp4_gemm import PackedNvfp4, decode_operand, pack_operand
from gleipnir.serving_precision import is_gdn_projection
from gleipnir.vllm_frost_fp4 import (
    FrostFp4Config,
    FrostFp4LinearMethod,
    is_mlp_projection,
)


class CheckedGdnFp4Method(FrostFp4LinearMethod):
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_gleipnir_frost_ready", False):
            return
        original = layer.weight
        super().process_weights_after_loading(layer)
        layer._gleipnir_gdn_kernel_checks = []
        if not re.search(r"\.layers\.0\.linear_attn\.", self.prefix):
            return
        weight = PackedNvfp4(layer.weight, layer.weight_scale, layer.weight_inverse)
        decoded = (
            decode_operand(
                PackedNvfp4(weight.codes[:16], weight.scales, weight.inverse)
            )
            .float()
            .cpu()
        )
        bn = decoded * weight.inverse.cpu()
        generator = torch.Generator(device=original.device).manual_seed(0)
        for m in (1, 17, 129):
            x = torch.randn(
                m,
                original.shape[1],
                device=original.device,
                dtype=torch.bfloat16,
                generator=generator,
            )
            activation = pack_operand(
                x, row_amax=True, chunked_rows=True, hardware_packing=True
            )
            an = (
                decode_operand(activation).float().cpu()
                * activation.inverse.cpu()[:, None]
            )
            reference = (
                (
                    (an @ bn.t()).to(torch.bfloat16).float()
                    / (activation.inverse.cpu()[:, None] * weight.inverse.cpu())
                )
                .to(torch.bfloat16)
                .float()
            )
            observed = self.apply(layer, x)[:, :16].float().cpu()
            relative = float(
                (observed - reference).norm() / reference.norm().clamp_min(1e-12)
            )
            if not bool(torch.isfinite(observed).all()) or relative > 0.01:
                raise ValueError(f"FROST GDN quantized-reference mismatch: {relative}")
            layer._gleipnir_gdn_kernel_checks.append(
                {
                    "rows": m,
                    "relative_l2": relative,
                    "weight_relative_l2": layer._gleipnir_weight_error,
                }
            )


@register_quantization_config("gleipnir_frost_gdn_fp4")
class FrostGdnFp4Config(FrostFp4Config):
    @classmethod
    def get_name(cls) -> str:
        return "gleipnir_frost_gdn_fp4"

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "FrostGdnFp4Config":
        return cls()

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if not isinstance(layer, LinearBase):
            return None
        if is_mlp_projection(prefix):
            return FrostFp4LinearMethod(prefix)
        if is_gdn_projection(prefix):
            return CheckedGdnFp4Method(prefix)
        return UnquantizedLinearMethod()
