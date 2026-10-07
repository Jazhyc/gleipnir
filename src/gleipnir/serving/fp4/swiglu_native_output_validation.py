"""Separate arithmetic admission from changed output-quantization precision."""

import math

from gleipnir.serving_fp4_swiglu_overhead_validation import ROWS

__all__ = ["validate_native", "validate_runtime"]


def validate_runtime(audit: dict, worker_pid: int, path: str) -> None:
    """Require native output on all MLPs and both precompiled row policies."""
    calls = [c for c in audit.get("calls", []) if c.get("backend") == "fused"]
    if (
        not audit.get("passed")
        or audit.get("worker_pid") != worker_pid
        or audit.get("validation_path") != path
        or audit.get("minimum_rows") != 4097
        or {c.get("layer") for c in calls} != set(range(32))
        or any(c.get("rows", 0) < 4097 for c in calls)
        or audit.get("compile_count") != 1
        or audit.get("overhead_compile_count") != 1
        or audit.get("generated_sha256") != audit.get("validated_generated_sha256")
    ):
        raise ValueError("native FP4 output live dispatch failed")


def validate_native(receipt: dict) -> None:
    """Require independent packing and decoded-GEMM checks plus changed replay."""
    records = receipt.get("results", [])
    if (
        receipt.get("state") != "completed"
        or not receipt.get("passed")
        or not receipt.get("weight_permutation_exact")
        or receipt.get("compile_count") != 1
        or receipt.get("scaling") != "unit_global_inverse_local_16_element_e4m3"
        or len(records) != len(ROWS)
        or {r.get("m") for r in records} != ROWS
    ):
        raise ValueError("incomplete direct FP4 output receipt")
    for r in records:
        if r.get("tile") != "native_block_fp4_output" or not all(
            r.get(k)
            for k in ("passed", "finite", "zero_row_exact", "unchanged_rows_exact")
        ):
            raise ValueError("failed direct FP4 output finite/isolation checks")
        for key in (
            "packing_reference",
            "arithmetic",
            "changed_replay",
            "packing_replay",
        ):
            v = r.get(key, {})
            error = v.get("relative_l2", math.inf)
            if (
                not v.get("passed")
                or not v.get("finite")
                or not math.isfinite(error)
                or error > 0.01
            ):
                raise ValueError("failed direct FP4 output arithmetic/replay")
