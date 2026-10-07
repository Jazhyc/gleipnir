"""Per-channel MLP FP8 plus explicit Triton block FP8 in other decoder linears."""

from __future__ import annotations

import os
import re
from typing import Any

import torch
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import init_fp8_linear_kernel
from vllm.model_executor.kernels.linear.scaled_mm.triton import (
    TritonFp8BlockScaledMMKernel,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PerBlockOnlineLinearMethod,
    Fp8PtpcOnlineLinearMethod,
)

logger = init_logger(__name__)


class TritonBlockFp8Method(Fp8PerBlockOnlineLinearMethod):
    """Use stock packing and arithmetic with a required per-layer Triton backend."""

    def create_weights(self, layer: torch.nn.Module, *args: Any, **kwargs: Any) -> None:
        super().create_weights(layer, *args, **kwargs)
        self.fp8_linear = init_fp8_linear_kernel(
            activation_quant_key=self.activation_quant_key,
            weight_quant_key=self.weight_quant_key,
            weight_shape=layer.weight.shape,
            input_dtype=self.input_dtype,
            out_dtype=self.out_dtype,
            force_kernel=TritonFp8BlockScaledMMKernel,
            module_name=self.__class__.__name__,
        )
        if not isinstance(self.fp8_linear, TritonFp8BlockScaledMMKernel):
            raise RuntimeError("Required Triton block FP8 kernel is unavailable")
        logger.info(
            "Required block FP8 final kernel: %s", type(self.fp8_linear).__name__
        )


@register_quantization_config("gleipnir_mixed_fp8")
class GleipnirMixedFp8Config(QuantizationConfig):
    """Keep the verified fast MLP recipe while testing other projection precision."""

    def __init__(self) -> None:
        super().__init__()
        self.scope = os.environ.get("GLEIPNIR_MIXED_FP8_SCOPE", "all")
        if self.scope not in {"all", "attention", "gdn"}:
            raise ValueError("Unsupported non-MLP mixed FP8 scope")

    @classmethod
    def get_name(cls) -> str:
        return "gleipnir_mixed_fp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 89

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> GleipnirMixedFp8Config:
        return cls()

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if not isinstance(layer, LinearBase):
            return None
        if not re.search(r"(?:^|\.)layers\.\d+\.", prefix):
            return UnquantizedLinearMethod()
        if ".mlp." in prefix:
            logger.info("Mixed precision channel FP8 MLP: %s", prefix)
            return Fp8PtpcOnlineLinearMethod()
        selected = self.scope == "all" or (
            ".self_attn." in prefix
            if self.scope == "attention"
            else ".linear_attn." in prefix
        )
        if selected:
            logger.info("Mixed precision Triton block FP8 decoder: %s", prefix)
            return TritonBlockFp8Method()
        return UnquantizedLinearMethod()
