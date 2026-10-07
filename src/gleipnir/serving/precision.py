"""Explicit decoder projection scope for mixed-precision serving experiments."""

import re
from typing import Any


def describe_attention_cache(value: Any) -> dict[str, Any]:
    """Audit a native cache tensor or separately strided K/V tensor pair."""
    if isinstance(value, tuple):
        if len(value) != 2 or str(value[0].dtype) != str(value[1].dtype):
            raise ValueError("invalid native K/V cache pair")
        return {
            "dtype": str(value[0].dtype),
            "shape": [list(t.shape) for t in value],
            "layout": "kv_pair",
        }
    return {"dtype": str(value.dtype), "shape": list(value.shape), "layout": "tensor"}


def is_gdn_projection(prefix: str) -> bool:
    """Select GDN QKV/Z and output GEMMs while retaining small gate projections."""
    if {"visual", "vision", "vision_model", "vision_encoder"} & set(prefix.split(".")):
        return False
    return (
        re.search(r"(?:^|\.)layers\.\d+\.linear_attn\.(in_proj_qkvz|out_proj)$", prefix)
        is not None
    )
