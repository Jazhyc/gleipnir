"""Experimental BF16 gate/up fusion without changing FP32 LoRA masters."""

from __future__ import annotations

from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F


def adapter_name(layer: torch.nn.Module) -> str:
    """Reject unsupported PEFT modes instead of silently changing semantics."""
    if (
        not hasattr(layer, "base_layer")
        or len(layer.active_adapters) != 1
        or layer.disable_adapters
        or layer.merged
        or layer.lora_variant
        or layer.base_layer.bias is not None
        or layer.base_layer.weight.requires_grad
        or layer.base_layer.weight.dtype != torch.bfloat16
    ):
        raise ValueError("MLP fusion requires a frozen BF16 standard single LoRA")
    name = layer.active_adapters[0]
    if name not in layer.lora_A or name not in layer.lora_B:
        raise ValueError("MLP fusion requires an active adapter in every projection")
    if (
        any(
            p.dtype != torch.float32
            for p in (layer.lora_A[name].weight, layer.lora_B[name].weight)
        )
        or getattr(layer.lora_dropout[name], "p", 0) != 0
        or layer.lora_A[name].bias is not None
        or layer.lora_B[name].bias is not None
    ):
        raise ValueError("MLP fusion requires FP32 bias-free masters and zero dropout")
    return name


def update(layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Retain PEFT's autocast, adapter projection and scaling boundaries."""
    name = layer.active_adapters[0]
    # Like PEFT: FP32 master input selection; matmuls follow caller autocast.
    return layer.lora_B[name](layer.lora_A[name](x.float())) * layer.scaling[name]


def merged_forward(self: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    base = F.linear(x, self._gleipnir_gate_up)
    gate, up = base.chunk(2, dim=-1)
    gate = (gate + update(self.gate_proj, x)).to(base.dtype)
    up = (up + update(self.up_proj, x)).to(base.dtype)
    return self.down_proj(F.silu(gate) * up)


def install_merged_mlp(
    model: torch.nn.Module, *, compile_mlp: bool = False
) -> dict[str, Any]:
    """Install only compatible MLP forwards; keep original parameter identities."""
    modules = [
        (name, m)
        for name, m in model.named_modules()
        if m.__class__.__name__ == "Qwen3_5MLP"
    ]
    if not modules:
        raise ValueError("no Qwen3.5 MLP modules found")
    for _, m in modules:
        if hasattr(m, "_gleipnir_gate_up"):
            raise ValueError("MLP fusion is already installed")
        for projection in (m.gate_proj, m.up_proj, m.down_proj):
            adapter_name(projection)
        if m.config.hidden_act != "silu":
            raise ValueError("MLP fusion requires SiLU")
    byte_count = 0
    for _, m in modules:
        combined = torch.cat((m.gate_proj.weight, m.up_proj.weight), dim=0).detach()
        m.register_buffer("_gleipnir_gate_up", combined, persistent=False)
        byte_count += combined.numel() * combined.element_size()
        fn = MethodType(merged_forward, m)
        m.forward = (
            torch.compile(fn, fullgraph=True, dynamic=True) if compile_mlp else fn
        )
    return {
        "modules": [name for name, _ in modules],
        "compiled": compile_mlp,
        "extra_frozen_buffer_bytes": byte_count,
        "master_parameters_preserved": True,
    }
