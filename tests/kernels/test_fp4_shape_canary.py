"""Decoder FP4 arithmetic checks must bind to observed model shapes."""

import json
from pathlib import Path

import pytest

from experiments.fp4_inference.shape_canary import SHAPES, validate_shapes


def summary():
    return {
        "unmatched_gemm_seconds": 0,
        "shapes": [{"n": n, "k": k, "projection": "linear_attn"} for n, k in SHAPES],
    }


def test_decoder_shapes_require_complete_observed_binding():
    validate_shapes(summary())
    changed = summary()
    changed["shapes"].pop()
    with pytest.raises(ValueError, match="shape binding"):
        validate_shapes(changed)
    changed = summary()
    changed["unmatched_gemm_seconds"] = 1
    with pytest.raises(ValueError, match="Unaccounted"):
        validate_shapes(changed)


def test_extra_or_non_decoder_shape_cannot_supply_binding():
    changed = summary()
    changed["shapes"].append({"n": 248320, "k": 2560, "projection": "lm_head"})
    validate_shapes(changed)
    changed["shapes"][0]["projection"] = "mlp.gate_up_proj"
    with pytest.raises(ValueError, match="shape binding"):
        validate_shapes(changed)


def test_wider_scope_preserves_original_kernel_and_scoring_choices():
    root = Path("experiments/fp4_inference/configs")
    original = json.loads((root / "nvfp4_mlp_cutlass.json").read_text())
    wider = json.loads((root / "nvfp4_all_cutlass.json").read_text())
    assert wider.pop("output") != original.pop("output")
    assert wider["environment"].pop("GLEIPNIR_NVFP4_SCOPE") == "all"
    assert original["environment"].pop("GLEIPNIR_NVFP4_SCOPE") == "mlp"
    assert wider == original
