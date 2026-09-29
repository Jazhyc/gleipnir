"""Experimental online NVFP4 MLPs using native vLLM/FlashInfer SM120 kernels.

Weight quantization uses one global FP32 scale and E4M3 scales per 16 values.
Activations use a dynamic global scale per invocation, avoiding calibration
leakage and clipping from an arbitrary static activation range.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
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
from vllm.model_executor.layers.quantization.online.fp8 import Fp8PtpcOnlineLinearMethod
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    cutlass_fp4_supported,
    swizzle_blockscale,
)
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.utils.flashinfer import (
    flashinfer_scaled_fp4_mm,
    has_flashinfer_b12x_gemm,
)

from gleipnir.nvfp4_artifact import NvFp4Artifact
from gleipnir.nvfp4_reference import decode_nvfp4
from gleipnir.vllm_fp32_logits import Fp32LogitsMethod

logger = init_logger(__name__)
NVFP4_MAX = 6.0 * 448.0


def pack_weight(
    weight: torch.Tensor,
    packer: str = "cuda",
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
    if packer == "cuda":
        packed, blocks = ops.scaled_fp4_quant(
            weight.contiguous(), scale.reciprocal(), is_sf_swizzled_layout=False
        )
    elif packer == "triton":
        from gleipnir.nvfp4_pack import pack_nvfp4

        packed, blocks = pack_nvfp4(weight.contiguous(), scale.reciprocal())
    else:
        raise ValueError(f"Unsupported NVFP4 packer: {packer}")
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
    packer: str = "cuda",
    scale_mode: str = "dynamic",
) -> torch.Tensor:
    """Include dynamic activation range, packing and native FP4 GEMM in the path."""
    original_shape = x.shape[:-1]
    matrix = x.reshape(-1, x.shape[-1]).contiguous()
    scale = matrix.abs().amax().float().clamp_min(1e-12).reshape(1) / NVFP4_MAX
    if scale_mode == "power2":
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    elif scale_mode != "dynamic":
        raise ValueError(f"Unsupported activation global scale: {scale_mode}")
    if packer == "cuda":
        packed, blocks = ops.scaled_fp4_quant(
            matrix, scale.reciprocal(), is_sf_swizzled_layout=True, backend=backend
        )
    elif packer == "triton":
        from gleipnir.nvfp4_pack import pack_nvfp4

        packed, blocks = pack_nvfp4(matrix, scale.reciprocal(), swizzled=True)
    else:
        raise ValueError(f"Unsupported NVFP4 packer: {packer}")
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

    def __init__(
        self,
        backend: str,
        prefix: str,
        packer: str,
        scale_mode: str,
        artifact: NvFp4Artifact | None = None,
    ) -> None:
        self.backend = backend
        self.prefix = prefix
        self.packer = packer
        self.scale_mode = scale_mode
        self.artifact = artifact
        if backend == "cutlass" and not cutlass_fp4_supported():
            raise RuntimeError("Native CUTLASS NVFP4 is unavailable")
        if backend == "b12x" and not has_flashinfer_b12x_gemm():
            raise RuntimeError("Native FlashInfer B12X NVFP4 is unavailable")

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_gleipnir_nvfp4_processed", False):
            return
        source = layer.weight
        packed, blocks, scale = (
            self.artifact.load(self.prefix, source)
            if self.artifact is not None
            else pack_weight(source, self.packer)
        )
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
            x,
            layer.weight,
            layer.weight_scale,
            layer.weight_global_scale,
            self.backend,
            self.packer,
            self.scale_mode,
        )
        return output if bias is None else output + bias


@register_quantization_config("gleipnir_nvfp4")
class GleipnirNvFp4Config(QuantizationConfig):
    """An explicit experiment method; standard inference remains unaffected."""

    def __init__(self) -> None:
        super().__init__()
        self.scope = os.environ.get("GLEIPNIR_NVFP4_SCOPE", "mlp")
        self.fp32_logits = os.environ.get("GLEIPNIR_NVFP4_FP32_LOGITS", "0")
        self.backend = os.environ.get("GLEIPNIR_NVFP4_BACKEND", "cutlass")
        self.packer = os.environ.get("GLEIPNIR_NVFP4_PACKER", "cuda")
        self.scale_mode = os.environ.get("GLEIPNIR_NVFP4_SCALE_MODE", "dynamic")
        self.projections = os.environ.get("GLEIPNIR_NVFP4_PROJECTIONS", "all")
        self.other_mlp_precision = os.environ.get(
            "GLEIPNIR_NVFP4_OTHER_MLP_PRECISION", "bf16"
        )
        self.keep_layers = {
            int(x)
            for x in os.environ.get("GLEIPNIR_NVFP4_KEEP_LAYERS", "").split(",")
            if x
        }
        if self.scope not in {"none", "mlp", "all"} or self.backend not in {
            "cutlass",
            "b12x",
        }:
            raise ValueError("Unsupported online NVFP4 scope/backend")
        if self.fp32_logits not in {"0", "1"}:
            raise ValueError("Unsupported FP32 logits flag")
        if self.packer not in {"cuda", "triton"} or self.scale_mode not in {
            "dynamic",
            "power2",
        }:
            raise ValueError("Unsupported online NVFP4 packer/scale")
        if self.projections not in {"all", "gate_up", "down"} or (
            self.other_mlp_precision not in {"bf16", "fp8_channel"}
        ):
            raise ValueError("Unsupported online NVFP4 projection/other precision")
        if self.scope == "all" and self.projections != "all":
            raise ValueError("Projection selection requires MLP scope")
        prepared = os.environ.get("GLEIPNIR_NVFP4_PREPARED", "")
        self.artifact = None
        if prepared:
            if self.scope != "mlp":
                raise ValueError("Prepared NVFP4 artifact requires MLP scope")
            self.artifact = NvFp4Artifact(
                Path(prepared),
                os.environ.get("GLEIPNIR_NVFP4_PREPARED_SHA256", ""),
                os.environ.get("GLEIPNIR_NVFP4_MERGE_SHA256", ""),
            )

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
        if isinstance(layer, VocabParallelEmbedding) and self.fp32_logits == "1":
            return Fp32LogitsMethod()
        if not isinstance(layer, LinearBase):
            return None
        index = re.search(r"(?:^|\.)layers\.(\d+)\.", prefix)
        keep = index is not None and int(index.group(1)) in self.keep_layers
        selected = (
            self.scope != "none"
            and index is not None
            and (".mlp." in prefix if self.scope == "mlp" else "lm_head" not in prefix)
        )
        projection_match = self.projections == "all" or prefix.endswith(
            f".{self.projections}_proj"
        )
        if keep:
            return UnquantizedLinearMethod()
        if not selected or not projection_match:
            if (
                index is not None
                and ".mlp." in prefix
                and self.other_mlp_precision == "fp8_channel"
            ):
                return Fp8PtpcOnlineLinearMethod()
            return UnquantizedLinearMethod()
        return NvFp4OnlineLinearMethod(
            self.backend, prefix, self.packer, self.scale_mode, self.artifact
        )
