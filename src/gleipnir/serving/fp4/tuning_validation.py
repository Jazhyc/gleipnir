"""Fail-closed numerical validation for a bounded FP4 tuning receipt."""

from gleipnir.serving_fp4_tuning import BASE_TILE, CANDIDATES, SHAPES


def validate_runtime_usage(audit: dict, worker_pid: int, validation_path: str) -> None:
    """Require live coverage of every projection and the declared selected tile."""
    calls = audit.get("calls", [])
    selected = audit.get("selected", {})
    if (
        not audit.get("passed")
        or audit.get("worker_pid") != worker_pid
        or audit.get("validation_path") != validation_path
        or set(selected) != set(SHAPES)
        or {c["shape"] for c in calls} != set(SHAPES)
        or any(c["tile"] != selected[c["shape"]][c["band"]] for c in calls)
    ):
        raise ValueError("native GEMM tuning usage audit failed")


def validate_selection(receipt: dict) -> dict:
    """Require all eight numerical/replay cases for every selected shape/tile."""
    import math

    selected = receipt.get("selected", {})
    if (
        not receipt.get("passed")
        or receipt.get("state") != "completed"
        or receipt.get("baseline_tile") != BASE_TILE
        or set(selected) != set(SHAPES)
    ):
        raise ValueError("incomplete native GEMM tuning receipt")
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
                or len(checks) != 8
                or {r["m"] for r in checks}
                != {1, 17, 129, 1536, 2304, 4096, 29184, 32768}
                or any(
                    not r.get("passed")
                    or not r.get("finite")
                    or not math.isfinite(r.get("relative_l2", math.inf))
                    or not r["relative_l2"] <= 0.01
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
