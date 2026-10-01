"""Frozen native NVFP4 MLP bases with higher-precision LoRA branches.

The published 1.0.5 linear backward assumes batch one and computes an unused
base-weight gradient. This wrapper flattens leading dimensions and computes
only input gradients for frozen bases, using the upstream quantizer and GEMM.
"""

from __future__ import annotations

import importlib.metadata
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import nn

FOUROVERSIX_VERSION = "1.0.5"
FOUROVERSIX_SDIST_SHA256 = (
    "51ab69c8c09e63d7575213f446f56bcbc9c36011ddca4c5d7b923710e1e6cb00"
)
MLP_PROJECTIONS = {"gate_proj", "up_proj", "down_proj"}


def mlp_loading_skip_patterns(precision: str) -> list[str] | None:
    """Use Transformers' full-name regex matching to retain original MLP weights."""
    if precision not in {"nf4", "bf16", "fouroversix"}:
        raise ValueError(f"unknown MLP precision: {precision}")
    return None if precision == "nf4" else [r".*\.mlp\..*", "lm_head"]


def is_mlp_base(name: str) -> bool:
    """Select decoder MLP bases, including PEFT's base_layer suffix."""
    parts = name.split(".")
    if parts[-1] == "base_layer":
        parts = parts[:-1]
    return len(parts) >= 2 and parts[-2] == "mlp" and parts[-1] in MLP_PROJECTIONS


@dataclass
class FrozenFp4Runtime:
    """Prepacked frozen weights and explicitly selected native operations."""

    weight: Any
    transposed_weight: Any
    activation_config: Any
    gradient_config: Any
    matmul: Callable
    forward_calls: int = 0
    backward_calls: int = 0
    dequantized_weight: torch.Tensor | None = None
    row_scaled_activations: bool = False


def normalize_activation_rows(
    inputs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unfused per-token normalization with a fixed quantizer global maximum of one."""
    scales = inputs.float().abs().amax(dim=-1, keepdim=True)
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    return (inputs.float() / scales).to(torch.bfloat16), scales


class FrozenFp4Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, runtime: FrozenFp4Runtime):
        ctx.runtime = runtime
        ctx.input_shape = inputs.shape
        ctx.input_dtype = inputs.dtype
        runtime.forward_calls += 1
        flattened = inputs.reshape(-1, inputs.shape[-1]).to(torch.bfloat16)
        scales = None
        if runtime.row_scaled_activations:
            flattened, scales = normalize_activation_rows(flattened)
        output = runtime.matmul(
            flattened,
            runtime.weight,
            input_config=runtime.activation_config,
        )
        if scales is not None:
            output = (output.float() * scales).to(torch.bfloat16)
        return output.reshape(*inputs.shape[:-1], output.shape[-1])

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor):
        runtime = ctx.runtime
        runtime.backward_calls += 1
        flattened = gradient.reshape(-1, gradient.shape[-1]).to(torch.bfloat16)
        if runtime.dequantized_weight is not None:
            # Frozen bases need dX only. Differentiate the forward's decoded
            # quantized weight, rather than an independently quantized transpose.
            result = flattened @ runtime.dequantized_weight
        else:
            result = runtime.matmul(
                flattened,
                runtime.transposed_weight,
                input_config=runtime.gradient_config,
            )
        return result.reshape(ctx.input_shape).to(ctx.input_dtype), None


class FrozenFourOverSixLinear(nn.Linear):
    """Preserve PEFT's standard linear interface and frozen BF16 master weight."""

    def __init__(self, original: nn.Linear, runtime: FrozenFp4Runtime):
        if original.weight.requires_grad or original.bias is not None:
            raise ValueError("native FP4 wrapper requires a frozen bias-free base")
        # Avoid initializing another large weight or consuming the adapter RNG.
        nn.Module.__init__(self)
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.weight = original.weight
        self.register_parameter("bias", None)
        self.runtime = runtime

    @torch.compiler.disable
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return FrozenFp4Function.apply(inputs, self.runtime)


def native_runtime(
    weight: torch.Tensor,
    *,
    backward_mode: str = "fp4",
    row_scaled_activations: bool = False,
) -> FrozenFp4Runtime:
    """Fail closed on package/hardware mismatch and prohibit reference fallback."""
    if importlib.metadata.version("fouroversix") != FOUROVERSIX_VERSION:
        raise RuntimeError("Four Over Six requires the isolated pinned 1.0.5 release")
    if backward_mode not in {"fp4", "dequantized_bf16"}:
        raise ValueError(f"unknown FP4 backward mode: {backward_mode}")
    if weight.device.type != "cuda" or torch.cuda.get_device_capability() not in {
        (10, 0),
        (10, 3),
    }:
        raise RuntimeError("native pilot requires B200/SM100 or B300/SM103 canaries")
    from fouroversix import ModuleQuantizationConfig, fp4_matmul, quantize_to_fp4
    from fouroversix.utils import MatmulBackend, QuantizeBackend

    config = ModuleQuantizationConfig(
        keep_master_weights=True,
        weight_scale_2d=True,
        scale_rule="mse",
        quantize_backend=QuantizeBackend.triton,
        matmul_backend=MatmulBackend.cutlass,
    )
    forward_weight = quantize_to_fp4(weight, config.get_weight_config())
    activation_config = config.get_activation_config()
    if row_scaled_activations:
        activation_config = replace(
            activation_config,
            kwargs={"x_amax": torch.ones(1, device=weight.device, dtype=torch.float32)},
        )
    backward_weight = (
        quantize_to_fp4(weight, config.get_weight_config(transpose=True))
        if backward_mode == "fp4"
        else None
    )
    decoded_weight = None
    if backward_mode == "dequantized_bf16":
        from fouroversix.quantize import dequantize

        # One-time exact packed-weight decoding; forward remains native CUTLASS.
        decoded_weight = dequantize(
            forward_weight,
            backend=QuantizeBackend.pytorch,
            dtype=torch.bfloat16,
            intermediate_dtype=torch.float32,
        ).contiguous()

    def matmul(inputs, packed_weight, *, input_config):
        return fp4_matmul(
            inputs,
            packed_weight,
            input_config=input_config,
            backend=MatmulBackend.cutlass,
            out_dtype=config.output_dtype,
        )

    return FrozenFp4Runtime(
        forward_weight,
        backward_weight,
        activation_config,
        config.get_gradient_config(),
        matmul,
        dequantized_weight=decoded_weight,
        row_scaled_activations=row_scaled_activations,
    )


def install_mlp_precision(
    model: nn.Module,
    precision: str,
    *,
    backward_mode: str = "fp4",
    row_scaled_activations: bool = False,
) -> dict[str, Any]:
    """Keep attention unchanged and convert only the frozen decoder MLP bases."""
    if precision not in {"nf4", "bf16", "fouroversix"}:
        raise ValueError(f"unknown MLP precision: {precision}")
    selected = [
        (name, module)
        for name, module in model.named_modules()
        if is_mlp_base(name) and not hasattr(module, "base_layer")
    ]
    if not selected:
        raise ValueError("no decoder MLP bases found")
    converted = []
    for name, module in selected:
        if precision == "nf4":
            if module.__class__.__name__ != "Linear4bit":
                raise ValueError(f"expected NF4 MLP base: {name}")
            continue
        if type(module) is not nn.Linear or module.weight.requires_grad:
            raise ValueError(f"expected a frozen dense MLP base: {name}")
        module.weight.data = module.weight.data.to(torch.bfloat16)
        if precision == "fouroversix":
            parent_name, attribute = name.rsplit(".", 1)
            setattr(
                model.get_submodule(parent_name),
                attribute,
                FrozenFourOverSixLinear(
                    module,
                    native_runtime(module.weight)
                    if backward_mode == "fp4" and not row_scaled_activations
                    else native_runtime(
                        module.weight,
                        backward_mode=backward_mode,
                        row_scaled_activations=row_scaled_activations,
                    ),
                ),
            )
        converted.append(name)
    return {
        "precision": precision,
        "modules": [name for name, _ in selected],
        "converted_modules": converted,
        "master_weight_source": "original_pinned_checkpoint",
        "master_weight_dtype": "nf4" if precision == "nf4" else "bfloat16",
        "fouroversix_version": FOUROVERSIX_VERSION
        if precision == "fouroversix"
        else None,
        "quantize_backend": "triton" if precision == "fouroversix" else None,
        "matmul_backend": "cutlass" if precision == "fouroversix" else None,
        "scale_rule": "mse_4_over_6" if precision == "fouroversix" else None,
        "weight_scale_2d": precision == "fouroversix",
        "gradient_rounding": "stochastic"
        if precision == "fouroversix" and backward_mode == "fp4"
        else None,
        "activation_scaling": (
            "per_token_unfused" if row_scaled_activations else "per_tensor"
        )
        if precision == "fouroversix"
        else None,
        "backward_mode": backward_mode if precision == "fouroversix" else "bf16",
        "backward_uses_forward_quantized_weight": (
            precision == "fouroversix" and backward_mode == "dequantized_bf16"
        ),
        "frozen_weight_gradient": False,
        "native_boundary_eager": precision == "fouroversix",
    }


def native_call_counts(model: nn.Module) -> dict[str, int]:
    layers = [m for m in model.modules() if isinstance(m, FrozenFourOverSixLinear)]
    return {
        "modules": len(layers),
        "forward": sum(m.runtime.forward_calls for m in layers),
        "backward": sum(m.runtime.backward_calls for m in layers),
        "fp4_backward": sum(
            m.runtime.backward_calls
            for m in layers
            if m.runtime.dequantized_weight is None
        ),
        "dequantized_bf16_backward": sum(
            m.runtime.backward_calls
            for m in layers
            if m.runtime.dequantized_weight is not None
        ),
    }


def install_eager_mlp_interfaces(model: nn.Module) -> list[str]:
    """Keep the complete MLP arithmetic outside compilation for a matched probe."""
    selected = [(n, m) for n, m in model.named_modules() if n.endswith(".mlp")]
    if not selected:
        raise ValueError("no decoder MLP interfaces found")
    for _, module in selected:
        module.forward = torch.compiler.disable(module.forward)
    return [name for name, _ in selected]
