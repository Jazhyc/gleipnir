"""Require complete numerical/replay coverage and live fused-MLP dispatch."""

import math

SELECTED = "tile=(256, 192),cluster=(2, 1),vector=True"
ROWS = {1, 17, 129, 1536, 2304, 4096, 29184, 32768}


def validate_native(receipt: dict) -> None:
    """Reject incomplete, nonfinite, failed or misidentified kernel receipts."""
    checks = [r for r in receipt.get("results", []) if r.get("tile") == SELECTED]
    if (
        receipt.get("state") != "completed"
        or not receipt.get("passed")
        or not receipt.get("full_envelope")
        or not receipt.get("weight_permutation_exact")
        or not receipt.get("n192_scale_fix")
        or SELECTED not in receipt.get("validated_candidates", [])
        or len(checks) != 8
        or {r["m"] for r in checks} != ROWS
    ):
        raise ValueError("incomplete native SwiGLU fusion receipt")
    for r in checks:
        if not all(
            r.get(key) for key in ("passed", "zero_row_exact", "unchanged_rows_exact")
        ):
            raise ValueError("failed native SwiGLU isolation/replay checks")
        for value in (r, r.get("activation", {}), r.get("changed_replay", {})):
            error = value.get("relative_l2", math.inf)
            if (
                not value.get("passed")
                or not value.get("finite")
                or not math.isfinite(error)
                or error > 0.01
            ):
                raise ValueError("failed native SwiGLU numerical checks")


def validate_runtime(audit: dict, worker_pid: int, validation_path: str) -> None:
    """Require actual fused calls from all 32 MLPs on the current worker."""
    fused = [c for c in audit.get("calls", []) if c.get("backend") == "fused"]
    if (
        not audit.get("passed")
        or audit.get("worker_pid") != worker_pid
        or audit.get("validation_path") != validation_path
        or audit.get("selected") != SELECTED
        or audit.get("minimum_rows") != 1536
        or {c.get("layer") for c in fused} != set(range(32))
        or any(c["rows"] < 1536 or c.get("selected") != SELECTED for c in fused)
        or audit.get("generated_sha256") != audit.get("validated_generated_sha256")
    ):
        raise ValueError("native SwiGLU runtime usage audit failed")
