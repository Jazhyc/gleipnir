import json

import pytest

from experiments.local_inference.summarize_gemm import recover_down


def fixture_data():
    row = {"candidate": "tile", "gpu_ms_per_call": 1.0, "passed": True}
    reference = {
        "measurements": [
            {"candidate": "torch_default", "gpu_ms_per_call": 1.1, "passed": False}
        ],
        "validation": {"tile": {"passed": True}, "torch_default": {"passed": False}},
    }
    return "down " + json.dumps(row), reference


def test_recovery_preserves_failed_reference():
    log, reference = fixture_data()
    result = recover_down(log, reference)
    assert result["measurements"][0]["speedup_vs_default"] == 1.1
    assert not result["measurements"][1]["passed"]


def test_recovery_rejects_duplicates():
    log, reference = fixture_data()
    with pytest.raises(ValueError, match="Duplicate"):
        recover_down(log + "\n" + log, reference)


def test_recovery_rejects_changed_validation():
    log, reference = fixture_data()
    reference["validation"]["tile"]["passed"] = False
    with pytest.raises(ValueError, match="Validation mismatch"):
        recover_down(log, reference)
