"""Forward-only vLLM integration of the validated training FROST FP4 arithmetic."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import torch
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)

from gleipnir.cudnn_fp4_epilogue import Nvfp4ScaledGemm
from gleipnir.cudnn_fp4_gemm import PackedNvfp4, decode_operand, pack_operand

_PLANS: dict[tuple[int, int, int], Nvfp4ScaledGemm] = {}
PLAN_ROWS = 16384


def is_mlp_projection(prefix: str) -> bool:
    """Select fused decoder MLP projections while preserving all other linears."""
    return (
        re.search(r"(?:^|\.)layers\.\d+\.mlp\.(gate_up_proj|down_proj)$", prefix)
        is not None
    )


def plan_key(device: int, k: int, n: int) -> tuple[int, int, int]:
    """Reuse symbolic-M native plans across every runtime activation row count."""
    return device, k, n


def runtime_receipt() -> dict[str, Any]:
    """Validate the training-forward source/runtime subset needed by inference."""
    from importlib.metadata import version

    import cudnn
    import triton
    from cudnn.gated_attention_block.kernels import proj_gemm

    from gleipnir.native_fp4_training import validate_kernel_sources

    validate_kernel_sources()
    digest = hashlib.sha256(Path(proj_gemm.__file__).read_bytes()).hexdigest()
    observed = {
        "cudnn_frontend": cudnn.__version__,
        "cudnn_backend": cudnn.backend_version(),
        "torch_cuda": torch.version.cuda,
        "triton": triton.__version__,
        "cutlass_dsl": version("nvidia-cutlass-dsl"),
        "proj_gemm_sha256": digest,
    }
    expected = {
        "cudnn_frontend": "1.31.0",
        "cudnn_backend": 92600,
        "torch_cuda": "13.0",
        "triton": "3.7.1",
        "cutlass_dsl": "4.8.0",
        "proj_gemm_sha256": (
            "7207e3da1894956bbac3cf5c0d491d142b924a1d0c5c6fcce35d3b748d8ba447"
        ),
    }
    mismatches = {k: v for k, v in observed.items() if v != expected[k]}
    if mismatches:
        raise ValueError(f"validated FROST training runtime changed: {mismatches}")
    return {
        "cudnn_frontend": cudnn.__version__,
        "cudnn_backend": cudnn.backend_version(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cutlass_dsl": version("nvidia-cutlass-dsl"),
        "plan_reference_rows": PLAN_ROWS,
        "proj_gemm_sha256": digest,
        "native_plan_count": len(_PLANS),
    }


@torch.library.custom_op("gleipnir::frost_inference_linear", mutates_args=())
def frost_inference_linear(
    x: torch.Tensor, codes: torch.Tensor, scales: torch.Tensor, inverse: torch.Tensor
) -> torch.Tensor:
    """Use the unchanged hardware row packing and fused runtime-M training GEMM."""
    matrix = x.reshape(-1, x.shape[-1]).contiguous()
    m, k = matrix.shape
    n = codes.shape[0]
    key = plan_key(x.device.index, k, n)
    if key not in _PLANS:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("warm FROST plan before CUDA graph capture")
        _PLANS[key] = Nvfp4ScaledGemm(PLAN_ROWS, k, n, runtime_m=True)
    activation = pack_operand(
        matrix, row_amax=True, chunked_rows=True, hardware_packing=True
    )
    output = _PLANS[key](activation, PackedNvfp4(codes, scales, inverse))
    return output.reshape(*x.shape[:-1], n)


@frost_inference_linear.register_fake
def _fake_linear(x, codes, scales, inverse):
    return x.new_empty((*x.shape[:-1], codes.shape[0]))


class FrostFp4LinearMethod(UnquantizedLinearMethod):
    """Pack only forward weights, then expose training arithmetic as a custom op."""

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_gleipnir_frost_ready", False):
            return
        original = layer.weight
        if not bool(torch.isfinite(original).all()):
            raise ValueError("nonfinite source weight")
        packed = pack_operand(original.contiguous(), weight=True)
        sample = PackedNvfp4(packed.codes[:16], packed.scales, packed.inverse)
        decoded = decode_operand(sample).float().cpu()
        source = original[:16].float().cpu()
        error = float((decoded - source).norm() / source.norm().clamp_min(1e-12))
        if not bool(torch.isfinite(decoded).all()) or error > 0.25:
            raise ValueError(f"FROST weight reconstruction error: {self.prefix}")
        layer.weight = torch.nn.Parameter(packed.codes, requires_grad=False)
        layer.register_parameter(
            "weight_scale", torch.nn.Parameter(packed.scales, requires_grad=False)
        )
        layer.register_parameter(
            "weight_inverse", torch.nn.Parameter(packed.inverse, requires_grad=False)
        )
        layer._gleipnir_frost_ready = True
        layer._gleipnir_weight_error = error
        layer._gleipnir_kernel_checks = []
        if re.search(r"\.layers\.0\.mlp\.", self.prefix):
            from gleipnir.cudnn_fp4_mlp import _native_linear

            zeros = torch.zeros(
                PLAN_ROWS,
                original.shape[1],
                device=original.device,
                dtype=torch.bfloat16,
            )
            _native_linear(zeros, original, None, False, True, True, True)
            del zeros
            generator = torch.Generator(device=original.device).manual_seed(0)
            for m in (1, 17, 129):
                x = torch.randn(
                    m,
                    original.shape[1],
                    device=original.device,
                    dtype=torch.bfloat16,
                    generator=generator,
                )
                a = pack_operand(
                    x, row_amax=True, chunked_rows=True, hardware_packing=True
                )
                # Preserve the training graph's raw BF16 rounding before descale.
                an = decode_operand(a).float().cpu() * a.inverse.cpu()[:, None]
                bn = decoded * packed.inverse.cpu()
                reference = (
                    (an @ bn.t()).to(torch.bfloat16).float()
                    / (a.inverse.cpu()[:, None] * packed.inverse.cpu())
                ).to(torch.bfloat16)
                observed = self.apply(layer, x)[:, :16].float().cpu()
                relative = float(
                    (observed - reference.float()).norm()
                    / reference.float().norm().clamp_min(1e-12)
                )
                if not bool(torch.isfinite(observed).all()) or relative > 0.01:
                    raise ValueError(
                        f"FROST quantized-reference mismatch: {self.prefix}"
                    )
                # Exercise the exact original training forward path on this weight.
                training_output = _native_linear(
                    x, original, None, False, True, True, True
                )
                if not torch.equal(self.apply(layer, x), training_output):
                    raise ValueError("FROST inference differs from training forward")
                layer._gleipnir_kernel_checks.append(
                    {
                        "rows": m,
                        "relative_l2": relative,
                        "training_forward_bitwise_equal": True,
                    }
                )

    def apply(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        output = frost_inference_linear(
            x, layer.weight, layer.weight_scale, layer.weight_inverse
        )
        return output if bias is None else output + bias


@register_quantization_config("gleipnir_frost_fp4")
class FrostFp4Config(QuantizationConfig):
    @classmethod
    def get_name(cls) -> str:
        return "gleipnir_frost_fp4"

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
    def from_config(cls, config: dict[str, Any]) -> FrostFp4Config:
        return cls()

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if not isinstance(layer, LinearBase):
            return None
        return (
            FrostFp4LinearMethod(prefix)
            if is_mlp_projection(prefix)
            else UnquantizedLinearMethod()
        )
