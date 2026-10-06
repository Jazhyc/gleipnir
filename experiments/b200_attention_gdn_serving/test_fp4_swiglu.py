"""Check the frozen-weight permutation's gate pairing and tile-scale lineage."""

from copy import deepcopy

import pytest

from gleipnir.serving_fp4_swiglu import adapt_source, interleaved_rows


@pytest.mark.parametrize("n", [64, 128, 18432])
def test_interleaving_preserves_gate_pair_and_scale_groups(n):
    rows = interleaved_rows(n)
    assert sorted(rows) == list(range(n))
    for block in range(n // 64):
        up, gate = (
            rows[block * 64 : block * 64 + 32],
            rows[block * 64 + 32 : (block + 1) * 64],
        )
        assert all(u == g + n // 2 for u, g in zip(up, gate, strict=True))
    # A packed weight's 16-row scale tile remains intact after permutation.
    for start in range(0, n, 16):
        group = rows[start : start + 16]
        assert group[0] % 16 == 0
        assert group == list(range(group[0], group[0] + 16))


@pytest.mark.parametrize("n", [0, -64, 32, 65])
def test_invalid_gate_width_rejected(n):
    with pytest.raises(ValueError, match="divisible by 64"):
        interleaved_rows(n)


def test_template_drift_rejected_before_codegen():
    with pytest.raises(ValueError, match="source drift"):
        adapt_source("unknown upstream kernel")


def native_receipt():
    from gleipnir.serving_fp4_swiglu_validation import ROWS, SELECTED

    return {
        "state": "completed",
        "passed": True,
        "full_envelope": True,
        "weight_permutation_exact": True,
        "n192_scale_fix": True,
        "validated_candidates": [SELECTED],
        "results": [
            {
                "m": m,
                "tile": SELECTED,
                "passed": True,
                "finite": True,
                "relative_l2": 0.0,
                "zero_row_exact": True,
                "unchanged_rows_exact": True,
                "activation": {"passed": True, "finite": True, "relative_l2": 0.0},
                "changed_replay": {"passed": True, "finite": True, "relative_l2": 0.0},
            }
            for m in ROWS
        ],
    }


def test_native_admission_rejects_incomplete_and_nonfinite_receipts():
    from gleipnir.serving_fp4_swiglu_validation import validate_native

    validate_native(native_receipt())
    for mutation in (
        {"finite": False},
        {"relative_l2": float("nan")},
        {"activation": {"passed": True, "finite": True, "relative_l2": 0.011}},
        {"changed_replay": {"passed": True, "finite": False}},
        {"unchanged_rows_exact": False},
        {"m": 32768},
    ):
        receipt = native_receipt()
        receipt["results"][0].update(mutation)
        with pytest.raises(ValueError):
            validate_native(receipt)
    receipt = native_receipt()
    receipt["n192_scale_fix"] = False
    with pytest.raises(ValueError):
        validate_native(receipt)


def test_runtime_rejects_stale_pid_missing_layer_wrong_kernel_and_policy():
    from gleipnir.serving_fp4_swiglu_validation import SELECTED, validate_runtime

    audit = {
        "passed": True,
        "worker_pid": 17,
        "validation_path": "receipt.json",
        "selected": SELECTED,
        "minimum_rows": 1536,
        "generated_sha256": "abc",
        "validated_generated_sha256": "abc",
        "calls": [
            {"layer": i, "rows": 32768, "backend": "fused", "selected": SELECTED}
            for i in range(32)
        ],
    }
    validate_runtime(audit, 17, "receipt.json")
    for mutation in (
        {"worker_pid": 18},
        {"calls": audit["calls"][:-1]},
        {"minimum_rows": 128},
        {"generated_sha256": "other"},
    ):
        with pytest.raises(ValueError):
            validate_runtime({**audit, **mutation}, 17, "receipt.json")
    other = deepcopy(audit)
    other["calls"][0]["rows"] = 17
    with pytest.raises(ValueError):
        validate_runtime(other, 17, "receipt.json")


def test_serving_condition_requires_fusion_receipt_and_audited_worker():
    import json
    from pathlib import Path

    from experiments.b200_attention_gdn_serving.run import resolve_condition

    condition = json.loads(
        Path(__file__).with_name("fp4_swiglu_fused.json").read_text()
    )
    assert resolve_condition(condition, {})["swiglu_fusion_validation"]
    for key in ("swiglu_fusion_validation", "gemm_reference_validation"):
        missing = {**condition}
        missing.pop(key)
        with pytest.raises(ValueError, match="SwiGLU fusion"):
            resolve_condition(missing, {})
