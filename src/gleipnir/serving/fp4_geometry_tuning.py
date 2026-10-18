"""Config-derived FP4 GEMM tile selection for Qwen3.5-family models.

The 4B tuning module (``gleipnir.serving.fp4.tuning``) is hash-bound and fixes
its four projection shapes. Its tile catalog, ``retile``, ``row_band`` and
``ShapeTunedGemm`` are shape-free and reused by import; this module derives
the four FROST FP4 projection shapes from the served text config and
validates a sweep receipt against them.
"""

from __future__ import annotations

import math
from typing import Any

from gleipnir.serving.fp4.tuning import BASE_TILE, CANDIDATES

ROWS = {1, 17, 129, 1536, 2304, 4096, 29184, 32768}


def fp4_shapes(text: Any) -> dict[str, tuple[int, int]]:
    """(K, N) of the merged MLP and large GDN projections that run FP4."""
    hidden, mlp = text.hidden_size, text.intermediate_size
    key = text.linear_num_key_heads * text.linear_key_head_dim
    value = text.linear_num_value_heads * text.linear_value_head_dim
    return {
        "mlp_gate_up": (hidden, 2 * mlp),
        "gdn_input": (hidden, 2 * key + 2 * value),
        "mlp_down": (mlp, hidden),
        "gdn_output": (value, hidden),
    }


def validate_selection(receipt: dict, shapes: dict[str, tuple[int, int]]) -> dict:
    """Require all eight numerical/replay cases for every selected shape/tile."""
    selected = receipt.get("selected", {})
    recorded = {name: tuple(dims) for name, dims in receipt.get("shapes", {}).items()}
    if (
        not receipt.get("passed")
        or receipt.get("state") != "completed"
        or receipt.get("baseline_tile") != BASE_TILE
        or recorded != shapes
        or set(selected) != set(shapes)
    ):
        raise ValueError("FP4 GEMM tuning receipt does not cover the served shapes")
    for name, bands in selected.items():
        if set(bands) != {"small", "large"}:
            raise ValueError("missing GEMM tuning row band")
        for tile in set(bands.values()):
            checks = [
                r
                for r in receipt["results"]
                if r["shape"] == name and r["tile"] == tile and "m" in r
            ]
            if (
                tile not in CANDIDATES
                or len(checks) != len(ROWS)
                or {r["m"] for r in checks} != ROWS
                or any(
                    not r.get("passed")
                    or not r.get("finite")
                    or not r.get("relative_l2", math.inf) <= 0.01
                    or not r.get("zero_row_exact")
                    or not r.get("unchanged_rows_exact")
                    or not r.get("changed_replay", {}).get("passed")
                    or not r["changed_replay"].get("finite")
                    or not r["changed_replay"].get("relative_l2", math.inf) <= 0.01
                    for r in checks
                )
            ):
                raise ValueError(f"invalid selected GEMM tile checks: {name}/{tile}")
    return selected
