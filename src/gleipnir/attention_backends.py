"""Explicit training attention selection and same-weight backend checks."""

from collections.abc import Callable
from importlib.metadata import version
from typing import Any


def attention_loader_kwargs(
    implementation: str | None, expected_version: str | None = None
) -> dict[str, str]:
    """Keep automatic selection by default; fail closed for pinned FA4."""
    if implementation is None:
        if expected_version is not None:
            raise ValueError("attention version requires an explicit backend")
        return {}
    if implementation not in {"sdpa", "flash_attention_4"}:
        raise ValueError(f"unsupported training attention backend: {implementation}")
    if implementation == "flash_attention_4":
        installed = version("flash-attn-4")
        if expected_version is None or installed != expected_version:
            raise ValueError(f"FA4 version drift: {installed} != {expected_version}")
    elif expected_version is not None:
        raise ValueError("SDPA uses the pinned Torch version, not flash-attn-4")
    return {"attn_implementation": implementation}


def compare_attention_backends(
    model: Any,
    *,
    reference: str,
    candidate: str,
    forward: Callable[[], Any],
    compare: Callable[[Any, Any], dict[str, Any]],
) -> dict[str, Any]:
    """Switch the same model for a bounded check and always restore its backend."""
    try:
        model.set_attn_implementation(reference)
        reference_output = forward()
        model.set_attn_implementation(candidate)
        candidate_output = forward()
        result = compare(reference_output, candidate_output)
        return {"reference": reference, "candidate": candidate, **result}
    finally:
        model.set_attn_implementation(candidate)


def install_eager_attention_interface(implementation: str) -> dict[str, str]:
    """Keep the external kernel router opaque while compiling surrounding layers."""
    import torch
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    if implementation not in {"sdpa", "flash_attention_4"}:
        raise ValueError("an eager attention interface requires an explicit backend")
    interface = ALL_ATTENTION_FUNCTIONS[implementation]
    ALL_ATTENTION_FUNCTIONS.register(implementation, torch.compiler.disable(interface))
    return {
        "implementation": implementation,
        "module": interface.__module__,
        "qualname": interface.__qualname__,
    }
