"""FROST MLPs plus checked native per-token/per-channel FP8 GDN projections."""

import re
from typing import Any

import torch
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.online.fp8 import Fp8PtpcOnlineLinearMethod

from gleipnir.serving_precision import is_gdn_projection
from gleipnir.vllm_frost_fp4 import (
    FrostFp4Config,
    FrostFp4LinearMethod,
    is_mlp_projection,
)


class CheckedGdnFp8Method(Fp8PtpcOnlineLinearMethod):
    def __init__(self, prefix: str) -> None:
        super().__init__()
        self.prefix = prefix

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        original = layer.weight
        n, k = original.shape
        if not bool(torch.isfinite(original).all()):
            raise ValueError("nonfinite GDN projection source weight")
        super().process_weights_after_loading(layer)
        if (
            layer.weight.dtype != torch.float8_e4m3fn
            or layer.weight.shape != (k, n)
            or not bool(torch.isfinite(layer.weight_scale).all())
            or not bool((layer.weight_scale > 0).all())
            or type(self.fp8_linear).__name__ != "CutlassFP8ScaledMMLinearKernel"
        ):
            raise ValueError("GDN projection did not select native W8A8 Cutlass")
        layer._gleipnir_gdn_kernel_checks = []
        if not re.search(r"\.layers\.0\.linear_attn\.", self.prefix):
            return
        codes = layer.weight[:, :16].float().cpu().t()
        scales = layer.weight_scale[:16].float().cpu().reshape(1, -1)
        decoded = codes * scales.t()
        source = original[:16].float().cpu()
        error = float((decoded - source).norm() / source.norm().clamp_min(1e-12))
        if error > 0.1:
            raise ValueError("FP8 GDN weight reconstruction error")
        generator = torch.Generator(device=original.device).manual_seed(0)
        for m in (1, 17, 129):
            x = torch.randn(
                m,
                k,
                device=original.device,
                dtype=torch.bfloat16,
                generator=generator,
            )
            a_scale = x.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 448
            a_codes = (x.float() / a_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
            reference = (
                ((a_codes.float().cpu() @ codes.t()) * a_scale.cpu() * scales)
                .to(torch.bfloat16)
                .float()
            )
            observed = self.apply(layer, x)[:, :16].float().cpu()
            relative = float(
                (observed - reference).norm() / reference.norm().clamp_min(1e-12)
            )
            if not bool(torch.isfinite(observed).all()) or relative > 0.01:
                raise ValueError(f"FP8 GDN quantized-reference mismatch: {relative}")
            layer._gleipnir_gdn_kernel_checks.append(
                {"rows": m, "relative_l2": relative, "weight_relative_l2": error}
            )


@register_quantization_config("gleipnir_frost_gdn")
class FrostGdnConfig(FrostFp4Config):
    @classmethod
    def get_name(cls) -> str:
        return "gleipnir_frost_gdn"

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "FrostGdnConfig":
        return cls()

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if not isinstance(layer, LinearBase):
            return None
        if is_mlp_projection(prefix):
            return FrostFp4LinearMethod(prefix)
        if is_gdn_projection(prefix):
            condition = get_current_vllm_config().additional_config["serving_condition"]
            if condition["gdn_projection_precision"] != "fp8":
                raise ValueError("unsupported GDN projection precision")
            return CheckedGdnFp8Method(prefix)
        return UnquantizedLinearMethod()
