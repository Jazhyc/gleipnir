"""Fail closed on incomplete symbolic-row validation and wrong serving scope."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from gleipnir.serving_fp4_swiglu_overhead_validation import ROWS, validate_native


def receipt():
    numerical = {"finite": True, "passed": True, "relative_l2": 0.0}
    return {
        "state": "completed",
        "passed": True,
        "weight_permutation_exact": True,
        "n192_scale_fix": True,
        "compile_count": 1,
        "results": [
            {
                "m": m,
                "tile": "symbolic_overhead",
                **numerical,
                "zero_row_exact": True,
                "unchanged_rows_exact": True,
                "activation": numerical.copy(),
                "changed_replay": numerical.copy(),
            }
            for m in sorted(ROWS)
        ],
    }


def test_native_requires_unaligned_rows_one_plan_and_complete_numerics():
    validate_native(receipt())
    for mutation in ({"compile_count": 2}, {"passed": False}, {"state": "failed"}):
        with pytest.raises(ValueError):
            validate_native({**receipt(), **mutation})
    for mutation in (
        {"m": 32768},
        {"relative_l2": float("nan")},
        {"activation": {"finite": True, "passed": True, "relative_l2": 0.011}},
        {"changed_replay": {"finite": False, "passed": True, "relative_l2": 0.0}},
        {"unchanged_rows_exact": False},
    ):
        r = deepcopy(receipt())
        r["results"][0].update(mutation)
        with pytest.raises(ValueError):
            validate_native(r)


def test_overhead_condition_retains_attention_fp4_and_requires_native_receipt():
    from experiments.b200_attention_gdn_serving.run import resolve_condition

    c = json.loads(Path(__file__).with_name("fp4_swiglu_overhead.json").read_text())
    assert resolve_condition(c, {})["swiglu_overhead_validation"]
    for key in ("swiglu_overhead_validation", "attention_projection_validation"):
        missing = {**c}
        missing.pop(key)
        with pytest.raises(ValueError):
            resolve_condition(missing, {})


def test_overhead_source_rejects_changed_pinned_inverse_epilogue(monkeypatch):
    from gleipnir import serving_fp4_swiglu_overhead as module

    monkeypatch.setattr(module, "adapt_source", lambda s: s)
    with pytest.raises(ValueError, match="source drift"):
        module.overhead_source("changed NVIDIA source")
