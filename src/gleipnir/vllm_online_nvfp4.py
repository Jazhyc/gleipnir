"""Online MLP packing with the pinned vLLM native NVFP4 inference kernel."""

from __future__ import annotations

import re
from typing import NamedTuple

import torch
from vllm import _custom_ops as ops
from vllm.model_executor.kernels.linear import init_nvfp4_linear_kernel
from vllm.model_executor.kernels.linear.nvfp4.flashinfer import (
    FlashInferCudnnNvFp4LinearKernel,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from gleipnir.nvfp4_reference import decode_nvfp4
from gleipnir.vllm_nvfp4 import NVFP4_MAX, pack_weight


def is_mlp_projection(prefix: str) -> bool:
    """Select only decoder gate/up and down projections, including fused loading."""
    return (
        re.search(r"(?:^|\.)layers\.\d+\.mlp\.(gate_up_proj|down_proj)$", prefix)
        is not None
    )


class _KernelLayer(NamedTuple):
    weight: torch.Tensor
    weight_scale: torch.Tensor
    input_global_scale_inv: torch.Tensor
    alpha: torch.Tensor
    output_size_per_partition: int
    weights_padding_cols: int


class OnlineNvFp4Method(UnquantizedLinearMethod):
    """Load BF16, then use stock packing/repacking/GEMM with dynamic input range."""

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.kernel = init_nvfp4_linear_kernel()
        if not isinstance(self.kernel, FlashInferCudnnNvFp4LinearKernel):
            raise ValueError("the B200 FP4 trial requires native FlashInfer cuDNN")

    def create_weights(self, layer, *args, **kwargs) -> None:
        super().create_weights(layer, *args, **kwargs)
        layer.output_size_per_partition = layer.weight.shape[0]

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_gleipnir_online_nvfp4", False):
            return
        original = layer.weight
        packed, blocks, global_scale = pack_weight(original, "cuda")
        decoded = torch.from_numpy(
            decode_nvfp4(
                packed[:16].cpu().numpy(),
                blocks[:16].float().cpu().numpy(),
                global_scale.item(),
            )
        )
        source = original[:16].float().cpu()
        error = float((decoded - source).norm() / source.norm().clamp_min(1e-12))
        if error > 0.25:
            raise ValueError(f"FP4 weight reconstruction error: {self.prefix}")
        layer.weight = torch.nn.Parameter(packed, requires_grad=False)
        layer.register_parameter(
            "weight_scale", torch.nn.Parameter(blocks, requires_grad=False)
        )
        layer.register_parameter(
            "weight_global_scale", torch.nn.Parameter(global_scale, requires_grad=False)
        )
        self.kernel.process_weights_after_loading(layer)
        layer._gleipnir_online_nvfp4 = True
        layer._gleipnir_fp4_weight_error = error
        layer._gleipnir_fp4_kernel_checks = []
        if re.search(r"\.layers\.0\.mlp\.", self.prefix):
            generator = torch.Generator(device=original.device).manual_seed(0)
            for rows in (1, 17, 65):
                x = torch.randn(
                    rows,
                    original.shape[1],
                    dtype=original.dtype,
                    device=original.device,
                    generator=generator,
                )
                scale = x.abs().amax().float().clamp_min(1e-12).reshape(1) / NVFP4_MAX
                codes, scales = ops.scaled_fp4_quant(x, scale.reciprocal(), False)
                x_decoded = torch.from_numpy(
                    decode_nvfp4(
                        codes.cpu().numpy(), scales.float().cpu().numpy(), scale.item()
                    )
                )
                reference = x_decoded @ decoded.t()
                observed = self.apply(layer, x)[:, :16].float().cpu()
                relative = float(
                    (observed - reference).norm() / reference.norm().clamp_min(1e-12)
                )
                if not bool(torch.isfinite(observed).all()) or relative > 0.01:
                    raise ValueError(
                        f"native FP4 decoded-reference mismatch: {self.prefix}"
                    )
                layer._gleipnir_fp4_kernel_checks.append(
                    {"rows": rows, "relative_l2": relative}
                )

    def apply(self, layer, x: torch.Tensor, bias=None) -> torch.Tensor:
        matrix = x.reshape(-1, x.shape[-1]).contiguous()
        scale = matrix.abs().amax().float().clamp_min(1e-12).reshape(1) / NVFP4_MAX
        view = _KernelLayer(
            layer.weight,
            layer.weight_scale,
            scale.reciprocal(),
            scale * layer.weight_global_scale,
            layer.output_size_per_partition,
            layer.weights_padding_cols,
        )
        output = self.kernel.apply_weights(view, matrix, bias)
        return output.reshape(*x.shape[:-1], layer.output_size_per_partition)


@register_quantization_config("gleipnir_b200_nvfp4")
class OnlineNvFp4Config(QuantizationConfig):
    @classmethod
    def get_name(cls) -> str:
        return "gleipnir_b200_nvfp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 100

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict) -> OnlineNvFp4Config:
        return cls()

    def get_quant_method(self, layer, prefix: str):
        if not isinstance(layer, LinearBase):
            return None
        return (
            OnlineNvFp4Method(prefix)
            if is_mlp_projection(prefix)
            else UnquantizedLinearMethod()
        )
