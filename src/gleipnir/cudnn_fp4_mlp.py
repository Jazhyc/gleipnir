"""Registered frozen-base NVFP4 forward/dgrad with original FP32 LoRA masters."""

from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F

from gleipnir.cudnn_fp4_gemm import Nvfp4Gemm, PackedNvfp4, pack_operand
from gleipnir.mlp_gemm import adapter_name, update


@dataclass
class FrozenPair:
    owners: tuple[torch.Tensor, ...]
    forward: PackedNvfp4
    backward: PackedNvfp4


_WEIGHTS: dict[tuple, FrozenPair] = {}
_PLANS: dict[tuple, Any] = {}


def _plan_key(device, m, k, n, *, fused_descale=False, runtime_m=False) -> tuple:
    if runtime_m and not fused_descale:
        raise ValueError("runtime M requires the fused descale launch path")
    key = (device, "runtime_m" if runtime_m else m, k, n)
    return (*key, "row_descale") if fused_descale else key


def clear_native_caches() -> None:
    """Release experiment-owned packed tensors/plans after all replays complete."""
    _PLANS.clear()
    _WEIGHTS.clear()


def cache_metadata() -> dict[str, int]:
    """Count native resident copies, excluding the original BF16 base parameters."""
    return {
        "weight_pairs": len(_WEIGHTS),
        "plans": len(_PLANS),
        "packed_weight_bytes": sum(
            t.numel() * t.element_size()
            for pair in _WEIGHTS.values()
            for operand in (pair.forward, pair.backward)
            for t in (operand.codes, operand.scales, operand.inverse)
        ),
    }


def prepare_weights(
    weight: torch.Tensor, other: torch.Tensor | None = None
) -> FrozenPair:
    """Prepare both orientations once; retain owners to prevent pointer reuse."""
    owners = (weight,) if other is None else (weight, other)
    if any(
        w.requires_grad or w.dtype != torch.bfloat16 or not w.is_cuda for w in owners
    ):
        raise ValueError("native FP4 requires frozen CUDA BF16 base weights")
    if any(w.ndim != 2 or w.shape[1] != weight.shape[1] for w in owners):
        raise ValueError("merged base weights must share their input width")
    key = tuple(
        (w.device.index, w.data_ptr(), w._version, tuple(w.shape)) for w in owners
    )
    if key not in _WEIGHTS:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("warm frozen FP4 weights before CUDA graph capture")
        merged = weight if other is None else torch.cat(owners, dim=0)
        with torch.no_grad():
            _WEIGHTS[key] = FrozenPair(
                owners,
                pack_operand(merged.contiguous(), weight=True),
                pack_operand(merged.t().contiguous(), weight=True),
            )
    return _WEIGHTS[key]


def _native_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    other: torch.Tensor | None,
    backward: bool,
    hardware_packing: bool = False,
    fused_descale: bool = False,
    runtime_m: bool = False,
) -> torch.Tensor:
    if not x.is_cuda or x.dtype != torch.bfloat16 or x.ndim < 2:
        raise ValueError("native FP4 MLP inputs must be CUDA BF16 with rank >= 2")
    with torch.cuda.device(x.device), torch.no_grad():
        pair = prepare_weights(weight, other)
        packed_weight = pair.backward if backward else pair.forward
        flat = x.reshape(-1, x.shape[-1]).contiguous()
        n = packed_weight.codes.shape[0]
        if flat.shape[1] != packed_weight.codes.shape[1] * 2:
            raise ValueError("FP4 projection input width mismatch")
        key = _plan_key(
            x.device.index,
            flat.shape[0],
            flat.shape[1],
            n,
            fused_descale=fused_descale,
            runtime_m=runtime_m,
        )
        if key not in _PLANS:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("warm every FP4 GEMM shape before graph capture")
            if fused_descale:
                from gleipnir.cudnn_fp4_epilogue import Nvfp4ScaledGemm

                _PLANS[key] = Nvfp4ScaledGemm(
                    flat.shape[0], flat.shape[1], n, runtime_m=runtime_m
                )
            else:
                _PLANS[key] = Nvfp4Gemm(flat.shape[0], flat.shape[1], n)
        packed_x = pack_operand(
            flat, row_amax=True, chunked_rows=True, hardware_packing=hardware_packing
        )
        return _PLANS[key](packed_x, packed_weight).reshape(*x.shape[:-1], n)


@torch.library.custom_op("gleipnir::fp4_frozen_linear", mutates_args=())
def fp4_frozen_linear(
    x: torch.Tensor, weight: torch.Tensor, other: torch.Tensor | None
) -> torch.Tensor:
    return _native_linear(x, weight, other, False)


@fp4_frozen_linear.register_fake
def _forward_fake(x, weight, other):
    n = weight.shape[0] + (0 if other is None else other.shape[0])
    return x.new_empty((*x.shape[:-1], n))


@torch.library.custom_op("gleipnir::fp4_frozen_dgrad", mutates_args=())
def fp4_frozen_dgrad(
    dy: torch.Tensor, weight: torch.Tensor, other: torch.Tensor | None
) -> torch.Tensor:
    return _native_linear(dy, weight, other, True)


@fp4_frozen_dgrad.register_fake
def _backward_fake(dy, weight, other):
    return dy.new_empty((*dy.shape[:-1], weight.shape[1]))


def _setup_context(ctx, inputs, output):
    _, weight, other = inputs
    ctx.has_other = other is not None
    ctx.save_for_backward(weight, *(() if other is None else (other,)))


def _backward(ctx, dy):
    saved = ctx.saved_tensors
    return (
        fp4_frozen_dgrad(dy, saved[0], saved[1] if ctx.has_other else None),
        None,
        None,
    )


fp4_frozen_linear.register_autograd(_backward, setup_context=_setup_context)


@torch.library.custom_op("gleipnir::fp4_hardware_linear", mutates_args=())
def fp4_hardware_linear(
    x: torch.Tensor, weight: torch.Tensor, other: torch.Tensor | None
) -> torch.Tensor:
    return _native_linear(x, weight, other, False, True)


fp4_hardware_linear.register_fake(_forward_fake)


@torch.library.custom_op("gleipnir::fp4_hardware_dgrad", mutates_args=())
def fp4_hardware_dgrad(
    dy: torch.Tensor, weight: torch.Tensor, other: torch.Tensor | None
) -> torch.Tensor:
    return _native_linear(dy, weight, other, True, True)


fp4_hardware_dgrad.register_fake(_backward_fake)


def _hardware_backward(ctx, dy):
    saved = ctx.saved_tensors
    return (
        fp4_hardware_dgrad(dy, saved[0], saved[1] if ctx.has_other else None),
        None,
        None,
    )


fp4_hardware_linear.register_autograd(_hardware_backward, setup_context=_setup_context)


@torch.library.custom_op("gleipnir::fp4_epilogue_linear", mutates_args=())
def fp4_epilogue_linear(
    x: torch.Tensor, weight: torch.Tensor, other: torch.Tensor | None
) -> torch.Tensor:
    return _native_linear(x, weight, other, False, True, True, True)


fp4_epilogue_linear.register_fake(_forward_fake)


@torch.library.custom_op("gleipnir::fp4_epilogue_dgrad", mutates_args=())
def fp4_epilogue_dgrad(
    dy: torch.Tensor, weight: torch.Tensor, other: torch.Tensor | None
) -> torch.Tensor:
    return _native_linear(dy, weight, other, True, True, True, True)


fp4_epilogue_dgrad.register_fake(_backward_fake)


def _epilogue_backward(ctx, dy):
    saved = ctx.saved_tensors
    return (
        fp4_epilogue_dgrad(dy, saved[0], saved[1] if ctx.has_other else None),
        None,
        None,
    )


fp4_epilogue_linear.register_autograd(_epilogue_backward, setup_context=_setup_context)


def fp4_mlp_forward(
    self: torch.nn.Module,
    x: torch.Tensor,
    *,
    hardware_packing: bool = False,
    fused_descale: bool = False,
) -> torch.Tensor:
    linear = fp4_hardware_linear if hardware_packing else fp4_frozen_linear
    if fused_descale:
        linear = fp4_epilogue_linear
    base = linear(
        x.to(torch.bfloat16),
        self.gate_proj.base_layer.weight,
        self.up_proj.base_layer.weight,
    )
    gate, up = base.chunk(2, dim=-1)
    gate = (gate + update(self.gate_proj, x)).to(base.dtype)
    up = (up + update(self.up_proj, x)).to(base.dtype)
    hidden = F.silu(gate) * up
    down = linear(hidden, self.down_proj.base_layer.weight, None)
    return (down + update(self.down_proj, hidden)).to(down.dtype)


def hardware_fp4_mlp_forward(self: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    return fp4_mlp_forward(self, x, hardware_packing=True)


def epilogue_fp4_mlp_forward(self: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    return fp4_mlp_forward(self, x, hardware_packing=True, fused_descale=True)


def install_fp4_mlp(
    model: torch.nn.Module,
    *,
    hardware_packing: bool = False,
    fused_descale: bool = False,
) -> dict[str, Any]:
    """Install before compiler wrapping; lazy CUDA packing follows Trainer placement."""
    if fused_descale and not hardware_packing:
        raise ValueError("fused descale requires the validated hardware packer")
    mlps = [
        (n, m) for n, m in model.named_modules() if m.__class__.__name__ == "Qwen3_5MLP"
    ]
    if not mlps:
        raise ValueError("no Qwen3.5 MLP modules found")
    for _, m in mlps:
        if getattr(m, "_gleipnir_fp4_installed", False):
            raise ValueError("native FP4 MLP is already installed")
        for projection in (m.gate_proj, m.up_proj, m.down_proj):
            adapter_name(projection)
            w = projection.base_layer.weight
            if any(d % 64 for d in w.shape):
                raise ValueError("native FP4 MLP dimensions must be divisible by 64")
        if m.config.hidden_act != "silu":
            raise ValueError("native FP4 MLP requires SiLU")
    for _, m in mlps:
        m.forward = MethodType(
            epilogue_fp4_mlp_forward
            if fused_descale
            else hardware_fp4_mlp_forward
            if hardware_packing
            else fp4_mlp_forward,
            m,
        )
        m._gleipnir_fp4_installed = True
    return {
        "modules": [n for n, _ in mlps],
        "base_forward": "nvfp4",
        "base_input_gradient": "nvfp4",
        "master_dtype": "float32",
        "activation_scale_scope": "row",
        "weight_scales": "16x16",
        "original_parameters_preserved": True,
        "lazy_cuda_preparation": True,
        "retained_merged_bf16_buffer": False,
        "hardware_packing": hardware_packing,
        "fused_descale": fused_descale,
        "runtime_m_plans": fused_descale,
    }
