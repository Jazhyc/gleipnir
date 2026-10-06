"""Require an explicit validated producer and preserve the accepted envelope."""

import json

import pytest

from experiments.b200_attention_gdn_serving.run import EXPERIMENT, resolve_condition
from experiments.b200_inference_benchmark.run import validate_preparation_usage


@pytest.mark.parametrize("mode", ["vendor", "silu", "norm", "combined"])
def test_preparation_conditions_preserve_precision_scope_and_reference(mode):
    condition = json.loads((EXPERIMENT / f"fp4_prepare_{mode}.json").read_text())
    resolved = resolve_condition(condition, {})
    assert resolved["fp4_preparation"] == mode
    assert resolved["quantization"] == "gleipnir_frost_gdn_fp4"
    assert resolved["attention_precision"] == "mxfp8"
    assert resolved["baseline"] == resolved["high_reference"] == "selected"
    assert resolved["serving_config_overrides"] == {
        "max_num_seqs": 128,
        "gpu_memory_utilization": 0.9,
    }
    assert "--kv-cache-dtype" not in resolved["extra_server_args"]
    with pytest.raises(ValueError, match="bound native receipt"):
        resolve_condition({**condition, "fp4_prepare_validation": None}, {})
    with pytest.raises(ValueError, match="bound native receipt"):
        resolve_condition({**condition, "fp4_preparation": "unvalidated"}, {})


def test_combined_preparation_cannot_reuse_only_one_native_receipt():
    condition = json.loads((EXPERIMENT / "fp4_prepare_combined.json").read_text())
    paths = condition["fp4_prepare_validation"]
    for value in (
        paths["vendor"],
        {"vendor": paths["vendor"]},
        {**paths, "norm": None},
    ):
        with pytest.raises(ValueError, match="all three native receipts"):
            resolve_condition({**condition, "fp4_prepare_validation": value}, {})


def test_preparation_usage_rejects_stale_and_partial_runtime_receipts():
    calls = [
        {"stage": stage}
        for stage, count in {"vendor": 48, "silu": 32, "norm": 32}.items()
        for _ in range(count)
    ]
    audit = {
        "passed": True,
        "worker_pid": 17,
        "condition": {"fp4_preparation": "combined"},
        "calls": calls,
    }
    validate_preparation_usage(audit, 17, "combined")
    for changed in (
        {**audit, "worker_pid": 18},
        {**audit, "passed": False},
        {**audit, "calls": calls[:-1]},
    ):
        with pytest.raises(ValueError, match="usage audit failed"):
            validate_preparation_usage(changed, 17, "combined")
    with pytest.raises(ValueError, match="usage audit failed"):
        validate_preparation_usage(audit, 17, "silu")
