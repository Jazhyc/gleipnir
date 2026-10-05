"""Scoped BF16 NVIDIA causal convolution for packed Qwen3.5 training."""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any

import torch

_BOUNDARIES: ContextVar[tuple | None] = ContextVar(
    "nvidia_conv_boundaries", default=None
)


def convolution_forward(original):
    """Carry existing packing offsets to the convolution without host reads."""

    @wraps(original)
    def forward(hidden_states, cache_params=None, attention_mask=None, **kwargs):
        if cache_params is not None or attention_mask is not None:
            raise ValueError(
                "NVIDIA convolution screen requires unpadded uncached rows"
            )
        cumulative = kwargs.get("cu_seq_lens_q")
        other = kwargs.get("cu_seq_lens_k")
        if cumulative is not None and other is not cumulative:
            raise ValueError("NVIDIA convolution requires shared Q/K offsets")
        sequence_ids = kwargs.get("seq_idx")
        if sequence_ids is not None and cumulative is None:
            raise ValueError("packed NVIDIA convolution requires cumulative offsets")
        token = _BOUNDARIES.set((sequence_ids, cumulative))
        try:
            return original(
                hidden_states, cache_params=None, attention_mask=None, **kwargs
            )
        finally:
            _BOUNDARIES.reset(token)

    if getattr(original, "_torchdynamo_disable", False):
        forward = torch.compiler.disable(forward)
    return forward


def convolution_kernel(
    backend,
    stats: dict,
    *,
    reference=None,
    diagnostics=None,
    name=None,
    return_reference: bool = False,
):
    """Reject unsupported routes rather than silently use dense convolution."""

    def convolution(x, weight, bias=None, activation=None, seq_idx=None):
        boundaries = _BOUNDARIES.get()
        if boundaries is None or boundaries[0] is not seq_idx:
            raise ValueError("convolution packing context missing or inconsistent")
        cumulative = boundaries[1]
        if not backend._can_route_causal_conv1d_bulk(
            x, weight, bias, cumulative, activation
        ):
            raise ValueError("unsupported NVIDIA native convolution layout or dtype")
        result = backend.causal_conv1d(
            x, weight, bias, activation, cu_seqlens=cumulative
        )
        route = backend._get_causal_conv1d_last_route()
        if route not in {"native-autograd", "native-inference"}:
            raise RuntimeError(f"unexpected NVIDIA convolution route: {route}")
        stats[route] = stats.get(route, 0) + 1
        if reference is not None:
            with torch.no_grad():
                expected = reference(
                    x=x,
                    weight=weight,
                    bias=bias,
                    activation=activation,
                    seq_idx=seq_idx,
                )
                delta = result.float() - expected.float()
                diagnostics.append(
                    {
                        "module": name,
                        "tokens": x.shape[-1],
                        "sequences": 1
                        if cumulative is None
                        else cumulative.numel() - 1,
                        "relative_l2": float(
                            delta.norm() / expected.float().norm().clamp_min(1e-30)
                        ),
                        "max_absolute": float(delta.abs().max()),
                        "input_min": float(x.min()),
                        "input_max": float(x.max()),
                    }
                )
            if return_reference:
                return reference(
                    x=x,
                    weight=weight,
                    bias=bias,
                    activation=activation,
                    seq_idx=seq_idx,
                )
        return result

    return convolution


@contextmanager
def nvidia_convolution_context(
    model: torch.nn.Module, *, diagnose: bool = False, return_reference: bool = False
) -> Iterator[dict[str, Any]]:
    """Replace all 24 convolutions, retaining parameters, hooks and GDN kernels."""
    backend = importlib.import_module("cudnn.ops.causal_conv1d")
    api = importlib.import_module("cudnn.causal_conv1d_bulk_sm100.api")
    modules = [
        (name, m)
        for name, m in model.named_modules()
        if m.__class__.__name__ == "Qwen3_5GatedDeltaNet"
    ]
    if len(modules) != 24:
        raise ValueError("NVIDIA convolution screen requires 24 Qwen3.5 GDN layers")
    for _, module in modules:
        weight = module.conv1d.weight
        if (
            weight.requires_grad
            or weight.dtype != torch.bfloat16
            or weight.ndim != 3
            or tuple(weight.shape[1:]) != (1, 4)
            or module.conv1d.bias is not None
            or module.activation != "silu"
            or module.causal_conv1d_fn is None
        ):
            raise ValueError(
                "NVIDIA convolution requires frozen biasless BF16 width-four SiLU"
            )
    stats: dict[str, int] = {"training_compiles": 0, "forward_compiles": 0}
    diagnostics = []
    compile_original = backend._compile_causal_conv1d_training_backend
    capacity = backend._CAUSAL_CONV1D_TRAINING_CACHE_CAPACITY
    forward_capacity = api._API_CACHE_CAPACITY
    forward_compile = api.CausalConv1dBulkFwdSm100.compile

    def compile_training(*args, **kwargs):
        stats["training_compiles"] += 1
        return compile_original(*args, **kwargs)

    def compile_forward(self):
        stats["forward_compiles"] += 1
        return forward_compile(self)

    missing = object()
    saved = [
        (m, m.__dict__.get("forward", missing), m.causal_conv1d_fn) for _, m in modules
    ]
    try:
        backend._compile_causal_conv1d_training_backend = compile_training
        # Twenty packed updates exceed the upstream defaults (64/128 plans).
        # Keep their exact-shape plans resident instead of cycling the LRU.
        backend._CAUSAL_CONV1D_TRAINING_CACHE_CAPACITY = max(capacity, 256)
        api._API_CACHE_CAPACITY = max(forward_capacity, 256)
        api.CausalConv1dBulkFwdSm100.compile = compile_forward
        for name, module in modules:
            original_kernel = module.causal_conv1d_fn
            module.forward = convolution_forward(module.forward)
            module.causal_conv1d_fn = convolution_kernel(
                backend,
                stats,
                reference=original_kernel if diagnose else None,
                diagnostics=diagnostics,
                name=name,
                return_reference=return_reference,
            )
        yield {
            "modules": [name for name, _ in modules],
            "stats": stats,
            "output_diagnostics": diagnostics,
            "backend": backend,
            "packing": "existing_cuda_int32_cumulative_offsets",
            "precision": "bf16",
            "parameter_identity_preserved": True,
        }
    finally:
        backend._compile_causal_conv1d_training_backend = compile_original
        backend._CAUSAL_CONV1D_TRAINING_CACHE_CAPACITY = capacity
        api._API_CACHE_CAPACITY = forward_capacity
        api.CausalConv1dBulkFwdSm100.compile = forward_compile
        for module, prior_forward, prior_kernel in saved:
            if prior_forward is missing:
                module.__dict__.pop("forward", None)
            else:
                module.forward = prior_forward
            module.causal_conv1d_fn = prior_kernel
