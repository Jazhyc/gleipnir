"""Fail closed on unmeasured shapes/rows and missing precompiled dispatch plans."""

import pytest

from gleipnir.serving_fp4_tuning import (
    BASE_TILE,
    ShapeTunedGemm,
    row_band,
    shape_key,
)
from gleipnir.serving_fp4_tuning_validation import (
    validate_runtime_usage,
    validate_selection,
)


def test_row_policy_rejects_unmeasured_envelope():
    assert row_band(1) == row_band(4096) == "small"
    assert row_band(4097) == row_band(32768) == "large"
    for rows in (0, -1, 32769):
        with pytest.raises(ValueError, match="outside validated envelope"):
            row_band(rows)
    assert shape_key(2560, 18432) == "mlp_gate_up"
    with pytest.raises(ValueError, match="unmeasured FP4 projection"):
        shape_key(2560, 4096)


def test_dispatch_requires_both_precompiled_row_bands():
    with pytest.raises(ValueError, match="both measured row bands"):
        ShapeTunedGemm({}, {"small": BASE_TILE})
    with pytest.raises(ValueError, match="no precompiled plan"):
        ShapeTunedGemm({}, {"small": BASE_TILE, "large": BASE_TILE})


def native_receipt():
    from gleipnir.serving_fp4_tuning import SHAPES

    return {
        "passed": True,
        "state": "completed",
        "baseline_tile": BASE_TILE,
        "selected": {name: {"small": BASE_TILE, "large": BASE_TILE} for name in SHAPES},
        "results": [
            {
                "shape": name,
                "tile": BASE_TILE,
                "m": m,
                "passed": True,
                "finite": True,
                "relative_l2": 0.0,
                "zero_row_exact": True,
                "unchanged_rows_exact": True,
                "changed_replay": {"passed": True, "finite": True, "relative_l2": 0.0},
            }
            for name in SHAPES
            for m in (1, 17, 129, 1536, 2304, 4096, 29184, 32768)
        ],
    }


def test_selection_rejects_missing_rows_and_nonfinite_or_failed_checks():
    assert validate_selection(native_receipt())
    for change in (
        {"relative_l2": float("nan")},
        {"relative_l2": 0.011},
        {"finite": False},
        {"unchanged_rows_exact": False},
        {"changed_replay": {"passed": True, "finite": False}},
        {"m": 17},
    ):
        receipt = native_receipt()
        receipt["results"][0].update(change)
        with pytest.raises(ValueError, match="invalid selected GEMM tile"):
            validate_selection(receipt)


def test_runtime_usage_rejects_stale_worker_or_wrong_tile():
    selected = native_receipt()["selected"]
    audit = {
        "passed": True,
        "worker_pid": 17,
        "validation_path": "receipt.json",
        "selected": selected,
        "calls": [
            {"shape": name, "band": "small", "tile": BASE_TILE} for name in selected
        ],
    }
    validate_runtime_usage(audit, 17, "receipt.json")
    for changed in (
        {**audit, "worker_pid": 18},
        {**audit, "validation_path": "other.json"},
        {**audit, "calls": audit["calls"][:-1]},
    ):
        with pytest.raises(ValueError, match="usage audit failed"):
            validate_runtime_usage(changed, 17, "receipt.json")
