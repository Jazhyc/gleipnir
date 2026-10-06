"""Check the independent FP4 packing reference and native source guards."""

import pytest
import torch

from gleipnir.serving_fp4_swiglu_block_reference import reference_decode


def test_reference_uses_even_ties_both_signs_and_exact_zero():
    value = torch.tensor(
        [
            [
                0.25,
                0.75,
                1.25,
                1.75,
                2.5,
                3.5,
                5.0,
                6.0,
                -0.25,
                -0.75,
                -1.25,
                -1.75,
                -2.5,
                -3.5,
                -5.0,
                -6.0,
            ]
        ]
    )
    expected = torch.tensor(
        [
            [
                0.0,
                1.0,
                1.0,
                2.0,
                2.0,
                4.0,
                4.0,
                6.0,
                0.0,
                -1.0,
                -1.0,
                -2.0,
                -2.0,
                -4.0,
                -4.0,
                -6.0,
            ]
        ]
    )
    assert torch.equal(reference_decode(value), expected)
    assert torch.equal(reference_decode(torch.zeros(2, 16)), torch.zeros(2, 16))


def test_reference_rejects_incomplete_block():
    with pytest.raises(ValueError, match="complete 16-value"):
        reference_decode(torch.zeros(1, 17))


def test_reference_models_finite_scale_and_value_saturation():
    value = torch.full((1, 16), 4800.0)
    assert torch.equal(reference_decode(value), torch.full_like(value, 2688.0))


def test_reference_scale_midpoint_rounds_to_even_before_fp4_values():
    value = torch.full((1, 16), 1.359375)
    value[0, 0] = 0.2734375
    result = reference_decode(value)
    assert result[0, 0] == 0.21875
    assert torch.equal(result[0, 1:], torch.full((15,), 1.3125))


def test_source_drift_rejected_before_native_codegen():
    from gleipnir.serving_fp4_swiglu_native_output import native_output_source

    with pytest.raises(ValueError, match="source drift"):
        native_output_source("changed NVIDIA kernel")


def test_native_arithmetic_admission_preserves_separate_precision_failure():
    from copy import deepcopy

    from gleipnir.serving_fp4_swiglu_native_output_validation import validate_native
    from gleipnir.serving_fp4_swiglu_overhead_validation import ROWS

    num = {"passed": True, "finite": True, "relative_l2": 0.0}
    receipt = {
        "state": "completed",
        "passed": True,
        "weight_permutation_exact": True,
        "compile_count": 1,
        "scaling": "unit_global_inverse_local_16_element_e4m3",
        "strict_baseline_precision_passed": False,
        "results": [
            {
                "m": m,
                "tile": "native_block_fp4_output",
                "passed": True,
                "finite": True,
                "zero_row_exact": True,
                "unchanged_rows_exact": True,
                **{
                    k: num.copy()
                    for k in (
                        "packing_reference",
                        "arithmetic",
                        "changed_replay",
                        "packing_replay",
                    )
                },
                "baseline_precision": {"passed": False, "relative_l2": 0.1},
            }
            for m in sorted(ROWS)
        ],
    }
    validate_native(receipt)
    assert receipt["strict_baseline_precision_passed"] is False
    for key in ("packing_reference", "arithmetic", "changed_replay", "packing_replay"):
        r = deepcopy(receipt)
        r["results"][0][key]["relative_l2"] = 0.011
        with pytest.raises(ValueError):
            validate_native(r)


def test_native_output_condition_requires_bound_recipe_scope():
    import json
    from pathlib import Path

    from experiments.b200_attention_gdn_serving.run import resolve_condition

    c = json.loads(
        Path(__file__).with_name("fp4_swiglu_native_output.json").read_text()
    )
    assert resolve_condition(c, {})["swiglu_native_output_validation"]
    c.pop("swiglu_native_output_validation")
    with pytest.raises(ValueError, match="native FP4 output"):
        resolve_condition(c, {})


def test_native_runtime_requires_large_row_dispatch_and_both_plans():
    from gleipnir.serving_fp4_swiglu_native_output_validation import validate_runtime

    audit = {
        "passed": True,
        "worker_pid": 17,
        "validation_path": "receipt.json",
        "minimum_rows": 4097,
        "compile_count": 1,
        "overhead_compile_count": 1,
        "generated_sha256": "hash",
        "validated_generated_sha256": "hash",
        "calls": [{"layer": i, "rows": 32768, "backend": "fused"} for i in range(32)],
    }
    validate_runtime(audit, 17, "receipt.json")
    for change in (
        {"minimum_rows": 1536},
        {"overhead_compile_count": 0},
        {"calls": audit["calls"][:-1]},
        {"worker_pid": 18},
    ):
        with pytest.raises(ValueError):
            validate_runtime({**audit, **change}, 17, "receipt.json")
