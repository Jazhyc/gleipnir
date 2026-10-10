"""Differentiable fixed residual projections for concept-ablation fine-tuning."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from gleipnir.data.monitoring import file_hash


def project_residual(hidden: torch.Tensor, unit: torch.Tensor) -> torch.Tensor:
    """Remove one component in FP32, preserving the autograd projection."""
    value = hidden.float()
    return (value - (value * unit).sum(-1, keepdim=True) * unit).to(hidden.dtype)


def install(model: nn.Module, path: Path, sha256: str) -> tuple[dict, list]:
    """Project decoder residuals during training; preserve state-dict keys."""
    if file_hash(path) != sha256:
        raise ValueError("concept direction archive drift")
    unit = np.load(path)["unit"]
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    layers = base.model.layers
    if (
        unit.shape != (base.config.hidden_size,)
        or not np.isfinite(unit).all()
        or not np.isclose(np.linalg.norm(unit), 1, atol=1e-6)
        or len(layers) != 32
    ):
        raise ValueError("invalid concept direction or Qwen geometry")
    device = next(model.parameters()).device
    direction = torch.from_numpy(unit.copy()).to(device=device, dtype=torch.float32)
    metadata = {
        "enabled": True,
        "archive_sha256": sha256,
        "direction_key": "unit",
        "direction_layer": 20,
        "applied_layers": list(range(32)),
        "positions": "all",
        "training_only": True,
        "forward_and_backward": True,
        "accumulation_dtype": "float32",
        "calls": {str(i): 0 for i in range(32)},
    }
    # A targeted operator check covers the changed computation, independently
    # of the preserved attention/packing/kernel startup receipts.
    probe = torch.arange(2 * 64 * len(unit), device=device, dtype=torch.float32)
    probe = (torch.sin(probe).reshape(2, 64, -1)).to(torch.bfloat16).requires_grad_()
    y = project_residual(probe, direction)
    grad = torch.autograd.grad(y.float().square().sum(), probe)[0].float()
    forward_error = float((y.detach().float() * direction).sum(-1).abs().max())
    backward_error = float((grad * direction).sum(-1).abs().max())
    forward_relative = float(
        (
            (y.detach().float() * direction).sum(-1).abs()
            / y.detach().float().norm(dim=-1).clamp_min(1e-10)
        ).max()
    )
    backward_relative = float(
        ((grad * direction).sum(-1).abs() / grad.norm(dim=-1).clamp_min(1e-10)).max()
    )
    upstream = (2 * y.detach().float()).to(torch.bfloat16).float()
    reference = (
        (upstream - (upstream * direction).sum(-1, keepdim=True) * direction)
        .to(torch.bfloat16)
        .float()
    )
    reference_error = float((grad - reference).abs().max())
    if (
        not torch.isfinite(y).all()
        or not torch.isfinite(grad).all()
        or max(forward_relative, backward_relative) > 0.001
        or reference_error > 0.0001
    ):
        raise ValueError("concept projection forward/backward canary failed")
    metadata["operator_canary"] = {
        "passed": True,
        "forward_alignment_max": forward_error,
        "backward_alignment_max": backward_error,
        "forward_relative_alignment_max": forward_relative,
        "backward_relative_alignment_max": backward_relative,
        "bf16_reference_gradient_max_difference": reference_error,
        "max_relative_alignment": 0.001,
        "max_reference_difference": 0.0001,
    }
    state_keys = tuple(model.state_dict())
    handles = []
    for index, layer in enumerate(layers):

        def hook(module, args, output, *, index=index):
            if not module.training:
                return output
            if not isinstance(output, torch.Tensor) or output.shape[-1] != len(unit):
                raise ValueError("decoder residual output layout changed")
            metadata["calls"][str(index)] += 1
            return project_residual(output, direction)

        handles.append(layer.register_forward_hook(hook))
    if tuple(model.state_dict()) != state_keys:
        raise ValueError("concept projection changed exported parameter identity")
    return metadata, handles
