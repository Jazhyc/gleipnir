"""Bounded decoder full-attention projection scope and native validation."""

import math
import re

SHAPES = {"qkv_proj": (2560, 10240), "o_proj": (4096, 2560)}
ROWS = {1, 17, 129, 1536, 2304, 4096, 29184, 32768}
EXPECTED = {(i, p) for i in range(3, 32, 4) for p in SHAPES}


def projection_identity(prefix: str) -> tuple[int, str] | None:
    """Select only pinned decoder attention layers, excluding vision modules."""
    if {"visual", "vision", "vision_model", "vision_encoder"} & set(prefix.split(".")):
        return None
    match = re.search(r"(?:^|\.)layers\.(\d+)\.self_attn\.(qkv_proj|o_proj)$", prefix)
    return (int(match[1]), match[2]) if match else None


def validate_native(receipt: dict) -> None:
    checks = receipt.get("checks", [])
    if (
        receipt.get("state") != "completed"
        or not receipt.get("passed")
        or len(checks) != 16
        or {(r["projection"], r["rows"]) for r in checks}
        != {(p, m) for p in SHAPES for m in ROWS}
    ):
        raise ValueError("incomplete FP4 attention projection validation")
    for r in checks:
        if tuple(r.get("shape", [])) != SHAPES[r["projection"]] or not all(
            r.get(k)
            for k in (
                "passed",
                "finite",
                "zero_row_exact",
                "unchanged_rows_exact",
                "replay_changed",
            )
        ):
            raise ValueError("failed FP4 attention projection isolation/scope")
        for e in (
            r.get("relative_l2", math.inf),
            r.get("replay_relative_l2", math.inf),
        ):
            if not math.isfinite(e) or e > 0.01:
                raise ValueError("failed FP4 attention projection arithmetic")


def validate_usage(audit: dict, worker_pid: int, validation_path: str) -> None:
    calls = audit.get("calls", [])
    if (
        not audit.get("passed")
        or audit.get("worker_pid") != worker_pid
        or audit.get("validation_path") != validation_path
        or {projection_identity(c["layer"]) for c in calls} != EXPECTED
        or any(
            tuple(c["shape"]) != SHAPES[projection_identity(c["layer"])[1]]
            or c["rows"] < 1
            or c["rows"] > 32768
            for c in calls
        )
    ):
        raise ValueError("native FP4 attention projection usage audit failed")
