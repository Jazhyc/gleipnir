"""Explicit decoder projection scope for mixed-precision serving experiments."""

import re


def is_gdn_projection(prefix: str) -> bool:
    """Select GDN QKV/Z and output GEMMs while retaining small gate projections."""
    if {"visual", "vision", "vision_model", "vision_encoder"} & set(prefix.split(".")):
        return False
    return (
        re.search(r"(?:^|\.)layers\.\d+\.linear_attn\.(in_proj_qkvz|out_proj)$", prefix)
        is not None
    )
