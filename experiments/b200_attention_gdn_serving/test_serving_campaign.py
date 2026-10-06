"""Frozen scoring contract and independently identified precision interventions."""

import json

import pytest

from experiments.b200_attention_gdn_serving.run import EXPERIMENT, resolve_condition


def test_capacity_only_control_does_not_quantize_attention():
    condition = json.loads((EXPERIMENT / "config.json").read_text())
    result = resolve_condition(condition, {"kernel": "first"})
    assert "--kv-cache-dtype" not in result["extra_server_args"]
    assert result["serving_config_overrides"]["max_num_seqs"] == 128
    assert result["high_concurrency"][-1] == 128
    assert result["startup_audit"].endswith("native_attention.json")


def test_fp8_counts_query_and_cache_scale_intervention_and_source_identity():
    condition = json.loads((EXPERIMENT / "config.json").read_text())
    condition["attention_precision"] = "fp8_e4m3"
    result = resolve_condition(condition, {"kernel": "first"})
    args = result["extra_server_args"]
    assert args[args.index("--kv-cache-dtype") + 1] == "fp8_e4m3"
    assert "--calculate-kv-scales" in args
    assert resolve_condition(condition, {"kernel": "second"}) != result


def test_precision_and_data_contract_changes_fail_closed():
    condition = json.loads((EXPERIMENT / "config.json").read_text())
    with pytest.raises(ValueError, match="attention precision"):
        resolve_condition({**condition, "attention_precision": "mxfp8"}, {})
    with pytest.raises(ValueError, match="frozen data/scoring"):
        resolve_condition({**condition, "serving_config_overrides": {"seed": 5}}, {})


def test_nvfp4_attention_uses_native_block_scale_cache_without_false_calibration():
    condition = json.loads((EXPERIMENT / "nvfp4_attention.json").read_text())
    args = resolve_condition(condition, {})["extra_server_args"]
    assert args[args.index("--kv-cache-dtype") + 1] == "nvfp4"
    assert "--calculate-kv-scales" not in args
    assert condition["gdn_projection_precision"] == "bf16"


def test_gdn_projection_intervention_keeps_attention_bf16_and_requires_its_quantizer():
    condition = json.loads((EXPERIMENT / "fp8_gdn_projection.json").read_text())
    result = resolve_condition(condition, {"gdn": "source"})
    assert "--kv-cache-dtype" not in result["extra_server_args"]
    assert result["gdn_projection_precision"] == "fp8"
    with pytest.raises(ValueError, match="mixed-precision quantizer"):
        resolve_condition({**condition, "quantization": "gleipnir_frost_fp4"}, {})
