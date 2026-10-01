"""Frozen native NVFP4 MLP bases with higher-precision LoRA branches.

The published 1.0.5 linear backward assumes batch one and computes an unused
base-weight gradient. This wrapper flattens leading dimensions and computes
only input gradients for frozen bases, using the upstream quantizer and GEMM.
"""

from __future__ import annotations

import importlib.metadata
import weakref
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


class PairedActivationCache:
    """Consume a gate projection's packed operand once, for the same up input.

    The weak input reference and tensor version reject changed or unrelated
    inputs. Every producer replaces the entry, every consumer clears it, and
    backward retains neither the entry nor the source activation.
    """

    def __init__(self) -> None:
        self.entry: tuple | None = None
        self.hits = 0
        self.misses = 0

    def produce(self, inputs: torch.Tensor, prepare: Callable) -> tuple:
        self.entry = None
        payload = prepare()
        if not inputs.is_inference():
            self.entry = (weakref.ref(inputs), inputs._version, payload)
        return payload

    def consume(self, inputs: torch.Tensor, prepare: Callable) -> tuple:
        entry, self.entry = self.entry, None
        if (
            entry is not None
            and entry[0]() is inputs
            and not inputs.is_inference()
            and entry[1] == inputs._version
        ):
            self.hits += 1
            return entry[2]
        self.misses += 1
        return prepare()


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
    observer: Callable | None = None
    fused_row_scaling: bool = False
    fused_activation_packing: bool = False
    activation_packer: Callable | None = None
    activation_cache: PairedActivationCache | None = None
    activation_cache_role: str | None = None
    activation_pack_calls: int = 0


def prepare_forward_activation(
    inputs: torch.Tensor, runtime: FrozenFp4Runtime
) -> tuple[Any, torch.Tensor | None]:
    """Prepare exact forward operands, optionally sharing gate/up packing."""

    def prepare():
        flattened = inputs.reshape(-1, inputs.shape[-1]).to(torch.bfloat16)
        scales = None
        if runtime.row_scaled_activations:
            if runtime.fused_activation_packing:
                from gleipnir.fp4_quantization_kernels import quantize_activation_rows

                flattened, scales = quantize_activation_rows(
                    flattened,
                    runtime.activation_config.kwargs["x_amax"],
                    implementation="tiled",
                )
            elif runtime.fused_row_scaling:
                from gleipnir.fp4_row_kernels import normalize_rows

                flattened, scales = normalize_rows(flattened)
            else:
                flattened, scales = normalize_activation_rows(flattened)
        if runtime.activation_cache is not None:
            runtime.activation_pack_calls += 1
            if not runtime.fused_activation_packing:
                flattened = runtime.activation_packer(
                    flattened, runtime.activation_config
                )
        return flattened, scales

    if runtime.activation_cache is None:
        return prepare()
    if runtime.observer is not None:
        raise ValueError("operand observation does not support shared packing")
    if runtime.activation_cache_role == "gate":
        return runtime.activation_cache.produce(inputs, prepare)
    if runtime.activation_cache_role == "up":
        return runtime.activation_cache.consume(inputs, prepare)
    raise ValueError("unknown shared activation role")


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
        if runtime.fused_activation_packing and runtime.observer is not None:
            raise ValueError("operand observation does not support fused packing")
        flattened, scales = prepare_forward_activation(inputs, runtime)
        output = runtime.matmul(
            flattened,
            runtime.weight,
            input_config=runtime.activation_config,
        )
        if scales is not None:
            if runtime.fused_row_scaling:
                from gleipnir.fp4_row_kernels import rescale_rows

                output = rescale_rows(output, scales)
            else:
                output = (output.float() * scales).to(torch.bfloat16)
        if runtime.observer is not None:
            runtime.observer(inputs, flattened, scales, output)
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
    fused_row_scaling: bool = False,
    fused_activation_packing: bool = False,
) -> FrozenFp4Runtime:
    """Fail closed on package/hardware mismatch and prohibit reference fallback."""
    if importlib.metadata.version("fouroversix") != FOUROVERSIX_VERSION:
        raise RuntimeError("Four Over Six requires the isolated pinned 1.0.5 release")
    if backward_mode not in {"fp4", "dequantized_bf16"}:
        raise ValueError(f"unknown FP4 backward mode: {backward_mode}")
    if fused_row_scaling and not row_scaled_activations:
        raise ValueError("fused row scaling requires per-token activations")
    if fused_activation_packing and not (
        row_scaled_activations
        and fused_row_scaling
        and backward_mode == "dequantized_bf16"
    ):
        raise ValueError(
            "fused packing requires fused per-token scaling and BF16 backward"
        )
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
        fused_row_scaling=fused_row_scaling,
        fused_activation_packing=fused_activation_packing,
        activation_packer=quantize_to_fp4,
    )


def install_mlp_precision(
    model: nn.Module,
    precision: str,
    *,
    backward_mode: str = "fp4",
    row_scaled_activations: bool = False,
    fused_row_scaling: bool = False,
    fused_activation_packing: bool = False,
    share_gate_up_activations: bool = False,
) -> dict[str, Any]:
    """Keep attention unchanged and convert only the frozen decoder MLP bases."""
    if precision not in {"nf4", "bf16", "fouroversix"}:
        raise ValueError(f"unknown MLP precision: {precision}")
    if fused_activation_packing and precision != "fouroversix":
        raise ValueError("fused activation packing requires native FP4 MLPs")
    if share_gate_up_activations and not (
        precision == "fouroversix"
        and row_scaled_activations
        and fused_row_scaling
        and backward_mode == "dequantized_bf16"
    ):
        raise ValueError("shared packing requires native fused per-token BF16 backward")
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
                    if backward_mode == "fp4"
                    and not row_scaled_activations
                    and not fused_row_scaling
                    and not fused_activation_packing
                    else native_runtime(
                        module.weight,
                        backward_mode=backward_mode,
                        row_scaled_activations=row_scaled_activations,
                        fused_row_scaling=fused_row_scaling,
                        fused_activation_packing=fused_activation_packing,
                    ),
                ),
            )
        converted.append(name)
    paired_modules = []
    if share_gate_up_activations:
        for name, module in model.named_modules():
            if not name.endswith(".mlp") and name != "mlp":
                continue
            pair = []
            for role in ["gate", "up"]:
                projection = getattr(module, f"{role}_proj", None)
                base = getattr(projection, "base_layer", projection)
                if not isinstance(base, FrozenFourOverSixLinear):
                    raise ValueError(f"missing native {role} projection in {name}")
                pair.append(base)
            cache = PairedActivationCache()
            for role, base in zip(["gate", "up"], pair, strict=True):
                base.runtime.activation_cache = cache
                base.runtime.activation_cache_role = role
            paired_modules.append(name)
        if not paired_modules:
            raise ValueError("no complete MLP gate/up pairs found")
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
            (
                "per_token_tiled_packing"
                if fused_activation_packing
                else "per_token_fused"
                if fused_row_scaling
                else "per_token_unfused"
            )
            if row_scaled_activations
            else "per_tensor"
        )
        if precision == "fouroversix"
        else None,
        "backward_mode": backward_mode if precision == "fouroversix" else "bf16",
        "backward_uses_forward_quantized_weight": (
            precision == "fouroversix" and backward_mode == "dequantized_bf16"
        ),
        "frozen_weight_gradient": False,
        "native_boundary_eager": precision == "fouroversix",
        "shared_gate_up_activation_modules": paired_modules,
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
        "shared_activation_pack_calls": sum(
            m.runtime.activation_pack_calls for m in layers
        ),
        "shared_activation_hits": sum(
            m.runtime.activation_cache.hits
            for m in layers
            if m.runtime.activation_cache_role == "gate"
        ),
        "shared_activation_misses": sum(
            m.runtime.activation_cache.misses
            for m in layers
            if m.runtime.activation_cache_role == "gate"
        ),
        "pending_shared_activation_entries": sum(
            int(m.runtime.activation_cache.entry is not None)
            for m in layers
            if m.runtime.activation_cache_role == "gate"
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


def install_eager_rmsnorm_interfaces(model: nn.Module) -> list[str]:
    """Retain native eager Qwen normalization at quantization-sensitive boundaries."""
    selected = [
        (name, module)
        for name, module in model.named_modules()
        if module.__class__.__name__ == "Qwen3_5RMSNorm"
    ]
    if not selected:
        raise ValueError("no Qwen3.5 RMSNorm interfaces found")
    for _, module in selected:
        module.forward = torch.compiler.disable(module.forward)
    return [name for name, _ in selected]


def install_eager_mlp_activation_interfaces(model: nn.Module) -> list[str]:
    """Preserve eager SiLU arithmetic at FP4 down-projection input boundaries."""
    selected = [
        (name, module)
        for name, module in model.named_modules()
        if name.endswith(".mlp.act_fn")
    ]
    if not selected:
        raise ValueError("no decoder MLP activation interfaces found")
    for _, module in selected:
        module.forward = torch.compiler.disable(module.forward)
    return [name for name, _ in selected]
