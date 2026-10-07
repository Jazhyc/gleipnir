"""Release GPU reference copies while retaining native forward/backward caches."""

from __future__ import annotations

import torch

from gleipnir.fouroversix_training import FrozenFourOverSixLinear


def offload_reference_weights(model: torch.nn.Module) -> dict[str, int | str]:
    """Move frozen original BF16 weights to CPU; never alter FP32 adapters."""
    layers = [
        module
        for module in model.modules()
        if isinstance(module, FrozenFourOverSixLinear)
    ]
    if not layers:
        raise ValueError("reference offload requires native FP4 bases")
    for layer in layers:
        if (
            layer.weight.requires_grad
            or layer.weight.dtype != torch.bfloat16
            or layer.runtime.dequantized_weight is None
            or layer.runtime.dequantized_weight.dtype != torch.bfloat16
            or not layer.runtime.dequantized_weight.is_cuda
            or not layer.runtime.weight.values.is_cuda
            or not layer.weight.is_cuda
        ):
            raise ValueError(
                "reference offload requires CUDA frozen BF16 references "
                "and native caches"
            )
    resident_bytes = sum(
        layer.weight.numel() * layer.weight.element_size()
        for layer in layers
        if layer.weight.is_cuda
    )
    for layer in layers:
        if layer.weight.is_cuda:
            layer.weight.data = layer.weight.detach().to("cpu")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return {
        "modules": len(layers),
        "released_reference_weight_bytes": resident_bytes,
        "reference_weight_device": "cpu",
        "forward_packed_weight_device": "cuda",
        "decoded_backward_weight_device": "cuda",
    }
