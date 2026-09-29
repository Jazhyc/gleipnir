"""Experimental online NVFP4 MLPs using native vLLM/FlashInfer SM120 kernels.

Weight quantization uses one global FP32 scale and E4M3 scales per 16 values.
Activations use a dynamic global scale per invocation, avoiding calibration
leakage and clipping from an arbitrary static activation range.
"""

from __future__ import annotations

import os
import re
from typing import Any

import torch
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    cutlass_fp4_supported,
    swizzle_blockscale,
)
from vllm.utils.flashinfer import (
    flashinfer_scaled_fp4_mm,
    has_flashinfer_b12x_gemm,
)

from gleipnir.nvfp4_reference import decode_nvfp4

logger = init_logger(__name__)
NVFP4_MAX = 6.0 * 448.0


def pack_weight(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack an unchanged BF16 weight, returning linear block scales and FP32 scale."""
    if weight.ndim != 2 or weight.shape[1] % 64 or weight.shape[0] % 32:
        raise ValueError(
            "Native NVFP4 requires a matrix with K divisible by 64, N by 32"
        )
    if weight.dtype not in (torch.bfloat16, torch.float16) or not weight.is_cuda:
        raise ValueError("NVFP4 packing requires CUDA BF16/FP16 weights")
    if not torch.isfinite(weight).all().item():
        raise ValueError("Nonfinite source weight")
    scale = weight.abs().amax().float().clamp_min(1e-12).reshape(1) / NVFP4_MAX
    packed, blocks = ops.scaled_fp4_quant(
        weight.contiguous(), scale.reciprocal(), is_sf_swizzled_layout=False
    )
    if blocks.shape != (weight.shape[0], weight.shape[1] // 16):
        raise ValueError("Native NVFP4 block-scale coverage mismatch")
    if not torch.isfinite(blocks.float()).all().item():
        raise ValueError("Nonfinite quantized weight scale")
    return packed, blocks, scale


def native_linear(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    swizzled_weight_scales: torch.Tensor,
    weight_global_scale: torch.Tensor,
    backend: str,
) -> torch.Tensor:
    """Include dynamic activation range, packing and native FP4 GEMM in the path."""
    original_shape = x.shape[:-1]
    matrix = x.reshape(-1, x.shape[-1]).contiguous()
    scale = matrix.abs().amax().float().clamp_min(1e-12).reshape(1) / NVFP4_MAX
    packed, blocks = ops.scaled_fp4_quant(
        matrix, scale.reciprocal(), is_sf_swizzled_layout=True, backend=backend
    )
    alpha = scale * weight_global_scale
    if backend == "cutlass":
        output = ops.cutlass_scaled_fp4_mm(
            packed, packed_weight, blocks, swizzled_weight_scales, alpha, x.dtype
        )
    elif backend == "b12x":
        output = flashinfer_scaled_fp4_mm(
            packed,
            packed_weight,
            blocks,
            swizzled_weight_scales,
            alpha,
            x.dtype,
            backend="b12x",
        )
    else:
        raise ValueError(f"Unsupported native NVFP4 backend: {backend}")
    return output.reshape(*original_shape, packed_weight.shape[0])


class NvFp4OnlineLinearMethod(UnquantizedLinearMethod):
    """Load standard BF16 checkpoint tensors and quantize selected linear layers."""

    def __init__(self, backend: str, prefix: str) -> None:
        self.backend = backend
        self.prefix = prefix
        if backend == "cutlass" and not cutlass_fp4_supported():
            raise RuntimeError("Native CUTLASS NVFP4 is unavailable")
        if backend == "b12x" and not has_flashinfer_b12x_gemm():
            raise RuntimeError("Native FlashInfer B12X NVFP4 is unavailable")

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_gleipnir_nvfp4_processed", False):
            return
        source = layer.weight
        packed, blocks, scale = pack_weight(source)
        sample_rows = min(source.shape[0], 16)
        decoded = decode_nvfp4(
            packed[:sample_rows].cpu().numpy(),
            blocks[:sample_rows].float().cpu().numpy(),
            scale.item(),
        )
        reconstructed = torch.from_numpy(decoded).to(source.device)
        sample = source[:sample_rows].float()
        relative = (
            (reconstructed - sample).norm() / sample.norm().clamp_min(1e-12)
        ).item()
        if relative > 0.25:
            raise ValueError(f"NVFP4 weight reconstruction exceeds 0.25: {self.prefix}")
        layer.weight = torch.nn.Parameter(packed, requires_grad=False)
        layer.register_parameter(
            "weight_scale",
            torch.nn.Parameter(swizzle_blockscale(blocks), requires_grad=False),
        )
        layer.register_parameter(
            "weight_global_scale", torch.nn.Parameter(scale, requires_grad=False)
        )
        layer._gleipnir_nvfp4_processed = True
        logger.info(
            "Native online NVFP4 backend=%s layer=%s shape=%s "
            "sample_weight_rel_l2=%.6f",
            self.backend,
            self.prefix,
            tuple(source.shape),
            relative,
        )

    def apply(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        output = native_linear(
            x, layer.weight, layer.weight_scale, layer.weight_global_scale, self.backend
        )
        return output if bias is None else output + bias


@register_quantization_config("gleipnir_nvfp4")
class GleipnirNvFp4Config(QuantizationConfig):
    """An explicit experiment method; standard inference remains unaffected."""

    def __init__(self) -> None:
        super().__init__()
        self.scope = os.environ.get("GLEIPNIR_NVFP4_SCOPE", "mlp")
        self.backend = os.environ.get("GLEIPNIR_NVFP4_BACKEND", "cutlass")
        self.keep_layers = {
            int(x)
            for x in os.environ.get("GLEIPNIR_NVFP4_KEEP_LAYERS", "").split(",")
            if x
        }
        if self.scope not in {"mlp", "all"} or self.backend not in {"cutlass", "b12x"}:
            raise ValueError("Unsupported online NVFP4 scope/backend")

    @classmethod
    def get_name(cls) -> str:
        return "gleipnir_nvfp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 120

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> GleipnirNvFp4Config:
        return cls()

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if not isinstance(layer, LinearBase):
            return None
        index = re.search(r"(?:^|\.)layers\.(\d+)\.", prefix)
        keep = index is not None and int(index.group(1)) in self.keep_layers
        selected = index is not None and (
            ".mlp." in prefix if self.scope == "mlp" else "lm_head" not in prefix
        )
        if keep or not selected:
            return UnquantizedLinearMethod()
        return NvFp4OnlineLinearMethod(self.backend, prefix)
