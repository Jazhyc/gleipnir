"""Explicit FP32 decision projection for bounded mixed-precision screens."""

from __future__ import annotations

from typing import Any


def install_fp32_lm_head(model: Any) -> dict[str, Any]:
    """Keep a frozen LM head outside outer autocast; preserve decoder precision."""
    import torch
    import torch.nn.functional as functional

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    head = base.lm_head
    if not isinstance(head, torch.nn.Linear) or any(
        p.requires_grad for p in head.parameters()
    ):
        raise ValueError("FP32 projection requires a frozen ordinary linear LM head")
    if hasattr(head, "gleipnir_fp32_projection"):
        return head.gleipnir_fp32_projection
    original = head.forward
    receipt = dict(
        weight_dtype=str(head.weight.dtype),
        projection_dtype="torch.float32",
        outer_autocast=False,
        original_output_dtypes=[],
        scope="LM-head selected positions; decoder/autocast and weights unchanged",
    )

    def forward(inputs):
        # Observe the actual original precision once, without keeping its graph.
        if not receipt["original_output_dtypes"]:
            with torch.no_grad():
                old = original(inputs)
                receipt["original_output_dtypes"].append(str(old.dtype))
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            return functional.linear(
                inputs.float(),
                head.weight.float(),
                None if head.bias is None else head.bias.float(),
            )

    head.forward = forward
    head.gleipnir_fp32_projection = receipt
    return receipt
