"""Geometry and receipt contract for the pinned 4B GDN FP8 intervention."""

import math
import re

from gleipnir.serving.precision import is_gdn_projection

SHAPES = {"in_proj_qkvz": (2560, 12288), "out_proj": (4096, 2560)}
ROWS = {1, 17, 129, 1536, 2304, 4096, 29184, 32768}
EXPECTED = {(i, p) for i in range(32) if (i + 1) % 4 for p in SHAPES}


def projection_identity(prefix: str) -> tuple[int, str] | None:
    """Identify only supported decoder GDN projections, excluding vision."""
    if not is_gdn_projection(prefix):
        return None
    match = re.search(r"(?:^|\.)layers\.(\d+)\.linear_attn\.(\w+)$", prefix)
    return int(match[1]), match[2]


def check_geometry(prefix: str, shape: tuple[int, ...]) -> None:
    """Reject unvalidated layers and shapes before quantizing weights."""
    identity = projection_identity(prefix)
    if identity not in EXPECTED or shape != tuple(reversed(SHAPES[identity[1]])):
        raise ValueError(f"unsupported GDN FP8 projection: {prefix}, {shape}")


def validate_native(receipt: dict) -> None:
    """Require every declared arithmetic, isolation and replay case."""
    checks = receipt.get("checks", [])
    if (
        not receipt.get("passed")
        or receipt.get("state") != "completed"
        or len(checks) != 16
        or {(c["projection"], c["rows"]) for c in checks}
        != {(p, m) for p in SHAPES for m in ROWS}
    ):
        raise ValueError("incomplete GDN FP8 native receipt")
    for c in checks:
        if (
            c["precision"] != "fp8"
            or tuple(c["shape"]) != SHAPES[c["projection"]]
            or not all(
                c.get(k)
                for k in (
                    "passed",
                    "finite",
                    "zero_row_exact",
                    "unchanged_rows_exact",
                    "replay_changed",
                )
            )
            or any(
                not math.isfinite(c[k]) or c[k] > 0.01
                for k in ("relative_l2", "replay_relative_l2")
            )
        ):
            raise ValueError("failed GDN FP8 native check")
