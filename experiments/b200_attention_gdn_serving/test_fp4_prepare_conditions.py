"""Require an explicit validated producer and preserve the accepted envelope."""

import json

import pytest

from experiments.b200_attention_gdn_serving.run import EXPERIMENT, resolve_condition


@pytest.mark.parametrize("mode", ["vendor", "silu", "norm"])
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
