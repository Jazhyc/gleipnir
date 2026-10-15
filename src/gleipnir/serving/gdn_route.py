"""Length-gated FlashInfer GDN prefill route for single-GPU serving.

FlashInfer's `use_cp="auto"` picks its context-parallel route from batch-head
parallelism alone, so a single short request takes the CP route (host planning
plus four kernels per layer). Below `min_tokens` scheduled tokens in a call,
this forces the published non-CP route; at or above it, FlashInfer's own
`"auto"` decision is left untouched, so long chunks and large batches keep
their exact existing arithmetic.
"""

from __future__ import annotations

import functools
import hashlib
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path
from typing import Any

FLASHINFER_VERSION = "0.7.0.post1"
FLASHINFER_SHA256 = "3dc5005eded5c9e152ebf32359e7a3238e0c0924522a1a3482f19fb8138b4f5f"

_COUNTS = {"non_cp": 0, "auto": 0}
_STATE = {"min_tokens": None}


def routed_call(original: Callable[..., Any], min_tokens: int, *args, **kwargs) -> Any:
    """Force non-CP below `min_tokens`; otherwise keep the caller's request."""
    if args or kwargs.get("backend") != "flashinfer":
        raise ValueError("GDN route gate requires keyword FlashInfer calls")
    if kwargs.get("use_cp", "auto") != "auto":
        raise ValueError("GDN route gate received an explicit use_cp")
    if kwargs["q"].shape[0] < min_tokens:
        kwargs["use_cp"] = False
        _COUNTS["non_cp"] += 1
    else:
        _COUNTS["auto"] += 1
    return original(**kwargs)


def install(min_tokens: int) -> None:
    from flashinfer import gdn_prefill

    if min_tokens < 1:
        raise ValueError("GDN route gate needs a positive token threshold")
    if version("flashinfer-python") != FLASHINFER_VERSION or (
        hashlib.sha256(Path(gdn_prefill.__file__).read_bytes()).hexdigest()
        != FLASHINFER_SHA256
    ):
        raise ValueError("FlashInfer GDN source changed")
    original = gdn_prefill.chunk_gated_delta_rule
    if getattr(original, "_gleipnir_nonparallel", False) or getattr(
        original, "_gleipnir_route_gate", None
    ):
        raise ValueError("another GDN route adapter is already installed")

    _STATE["min_tokens"] = min_tokens

    @functools.wraps(original)
    def gated(*args, **kwargs):
        return routed_call(original, _STATE["min_tokens"], *args, **kwargs)

    gated._gleipnir_route_gate = True
    gdn_prefill.chunk_gated_delta_rule = gated


def set_threshold(min_tokens: int) -> None:
    """Diagnostic paired A/B within one server; takes effect at the next call."""
    if _STATE["min_tokens"] is None or min_tokens < 1:
        raise ValueError("GDN route gate is not installed or threshold invalid")
    _STATE["min_tokens"] = min_tokens


def audit() -> dict:
    return {**_COUNTS, "min_tokens": _STATE["min_tokens"]}
