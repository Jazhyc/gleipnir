"""Admission and dispatch checks for the symbolic-row SwiGLU producer."""

import math

ROWS = {1, 17, 129, 1536, 1537, 2304, 4096, 29184, 32768}


def validate_native(receipt: dict) -> None:
    """Require the declared envelope and one plan across changed row counts."""
    records = receipt.get("results", [])
    if (
        receipt.get("state") != "completed"
        or not receipt.get("passed")
        or not receipt.get("weight_permutation_exact")
        or not receipt.get("n192_scale_fix")
        or receipt.get("compile_count") != 1
        or len(records) != len(ROWS)
        or {r.get("m") for r in records} != ROWS
    ):
        raise ValueError("incomplete symbolic SwiGLU receipt")
    for r in records:
        if r.get("tile") != "symbolic_overhead" or not all(
            r.get(k) for k in ("passed", "zero_row_exact", "unchanged_rows_exact")
        ):
            raise ValueError("failed symbolic SwiGLU isolation/dispatch")
        for v in (r, r.get("activation", {}), r.get("changed_replay", {})):
            error = v.get("relative_l2", math.inf)
            if (
                not v.get("finite")
                or not v.get("passed")
                or not math.isfinite(error)
                or error > 0.01
            ):
                raise ValueError("failed symbolic SwiGLU arithmetic/replay")


def validate_runtime(audit: dict, worker_pid: int, path: str) -> None:
    """Require actual calls from every MLP and the validated kernel identity."""
    calls = [c for c in audit.get("calls", []) if c.get("backend") == "fused"]
    if (
        not audit.get("passed")
        or audit.get("worker_pid") != worker_pid
        or audit.get("validation_path") != path
        or audit.get("minimum_rows") != 1536
        or {c.get("layer") for c in calls} != set(range(32))
        or any(c.get("rows", 0) < 1536 for c in calls)
        or audit.get("compile_count") != 1
        or audit.get("generated_sha256") != audit.get("validated_generated_sha256")
    ):
        raise ValueError("symbolic SwiGLU live dispatch failed")
