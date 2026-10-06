"""Fail closed on precision scope, incomplete native checks and stale usage."""

import json
from pathlib import Path

import pytest

from gleipnir.serving_attention_fp4 import (
    EXPECTED,
    ROWS,
    SHAPES,
    projection_identity,
    validate_native,
    validate_usage,
)


def test_projection_scope_excludes_vision_small_gates_head_and_other_linears():
    assert projection_identity("language_model.model.layers.3.self_attn.qkv_proj") == (
        3,
        "qkv_proj",
    )
    assert projection_identity("model.layers.31.self_attn.o_proj") == (31, "o_proj")
    for name in [
        "vision_model.layers.3.self_attn.qkv_proj",
        "model.layers.0.linear_attn.in_proj_ba",
        "lm_head",
        "model.layers.3.self_attn.q_norm",
        "model.layers.3.mlp.down_proj",
    ]:
        assert projection_identity(name) is None


def receipt():
    return {
        "state": "completed",
        "passed": True,
        "checks": [
            {
                "projection": p,
                "rows": m,
                "shape": list(SHAPES[p]),
                "passed": True,
                "finite": True,
                "zero_row_exact": True,
                "unchanged_rows_exact": True,
                "replay_changed": True,
                "relative_l2": 0.0,
                "replay_relative_l2": 0.0,
            }
            for p in SHAPES
            for m in sorted(ROWS)
        ],
    }


def test_native_receipt_rejects_missing_duplicate_nonfinite_wrong_shape_and_replay():
    validate_native(receipt())
    for mutation in [
        {"rows": 17},
        {"finite": False},
        {"relative_l2": float("nan")},
        {"replay_relative_l2": 0.011},
        {"shape": [2560, 18432]},
        {"unchanged_rows_exact": False},
    ]:
        r = receipt()
        r["checks"][0].update(mutation)
        with pytest.raises(ValueError):
            validate_native(r)


def test_usage_requires_current_worker_all_16_modules_and_geometry():
    calls = [
        {
            "layer": f"model.layers.{i}.self_attn.{p}",
            "rows": 32768,
            "shape": list(SHAPES[p]),
        }
        for i, p in sorted(EXPECTED)
    ]
    audit = {
        "passed": True,
        "worker_pid": 17,
        "validation_path": "receipt.json",
        "calls": calls,
    }
    validate_usage(audit, 17, "receipt.json")
    for mutation in [
        {"worker_pid": 18},
        {"calls": calls[:-1]},
        {"validation_path": "other.json"},
    ]:
        with pytest.raises(ValueError):
            validate_usage({**audit, **mutation}, 17, "receipt.json")


def test_condition_requires_bound_attention_worker_precision_and_quantizer():
    from experiments.b200_attention_gdn_serving.run import resolve_condition

    condition = json.loads(Path(__file__).with_name("attention_fp4.json").read_text())
    resolved = resolve_condition(condition, {})
    assert resolved["baseline"] == resolved["high_reference"] == "selected"
    assert resolved["attention_precision"] == "mxfp8"
    for key in ["attention_projection_validation", "attention_projection_precision"]:
        r = {**condition}
        r.pop(key)
        with pytest.raises(ValueError):
            resolve_condition(r, {})
    with pytest.raises(ValueError):
        resolve_condition({**condition, "quantization": "gleipnir_frost_gdn_fp4"}, {})


def test_combined_preparation_counts_additional_projection_packers():
    from experiments.b200_inference_benchmark.run import validate_preparation_usage

    calls = [
        {"stage": stage}
        for stage, n in {"vendor": 64, "norm": 32, "silu": 32}.items()
        for _ in range(n)
    ]
    audit = {
        "passed": True,
        "worker_pid": 17,
        "condition": {
            "fp4_preparation": "combined",
            "attention_projection_precision": "fp4",
        },
        "calls": calls,
    }
    validate_preparation_usage(audit, 17, "combined")
    with pytest.raises(ValueError):
        validate_preparation_usage({**audit, "calls": calls[:-1]}, 17, "combined")
