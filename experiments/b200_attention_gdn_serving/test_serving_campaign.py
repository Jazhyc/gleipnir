"""Frozen scoring contract and independently identified precision interventions."""

import json

import pytest
import yaml

from experiments.b200_attention_gdn_serving.run import EXPERIMENT, resolve_condition
from experiments.b200_inference_benchmark.run import server_command


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
        resolve_condition({**condition, "attention_precision": "fp6"}, {})
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


def test_alternative_gdn_backend_requires_a_bound_native_validation():
    condition = json.loads((EXPERIMENT / "config.json").read_text())
    with pytest.raises(ValueError, match="validation and audited worker"):
        resolve_condition({**condition, "gdn_backend": "cutedsl"}, {})
    result = resolve_condition(
        {
            **condition,
            "gdn_backend": "cutedsl",
            "gdn_validation": "canary.json",
            "worker_cls": "experiments.b200_attention_gdn_serving.gdn_worker."
            "GdnServingAuditWorker",
        },
        {},
    )
    assert result["gdn_backend"] == "cutedsl"


def test_server_launch_has_one_explicit_gdn_backend_and_preserves_default():
    config = yaml.safe_load(
        (EXPERIMENT.parent / "b200_inference_benchmark/config.yaml").read_text()
    )
    for backend in (None, "cutedsl"):
        selected = (
            config if backend is None else {**config, "gdn_prefill_backend": backend}
        )
        command = server_command(selected)
        assert command.count("--gdn-prefill-backend") == 1
        assert command[command.index("--gdn-prefill-backend") + 1] == (
            backend or "flashinfer"
        )


def test_blackwell_fa4_needs_its_native_audit_and_does_not_claim_fp8_support():
    condition = json.loads((EXPERIMENT / "fa4_attention.json").read_text())
    args = resolve_condition(condition, {})["extra_server_args"]
    selected = json.loads(args[args.index("--attention-config") + 1])
    assert selected == {"backend": "FLASH_ATTN", "flash_attn_version": 4}
    assert "--attention-backend" not in args
    with pytest.raises(ValueError, match="Blackwell FA4 requires BF16"):
        resolve_condition({**condition, "attention_precision": "fp8_e4m3"}, {})


def test_fp4_gdn_trial_cannot_launch_the_fp8_quantizer_under_an_fp4_label():
    condition = json.loads((EXPERIMENT / "fp4_gdn_projection.json").read_text())
    result = resolve_condition(condition, {})
    assert result["gdn_projection_precision"] == "fp4"
    assert result["quantization"] == "gleipnir_frost_gdn_fp4"
    with pytest.raises(ValueError, match="matching mixed-precision"):
        resolve_condition({**condition, "quantization": "gleipnir_frost_gdn"}, {})


def test_further_backends_retain_the_user_selected_fp4_gdn_scope():
    for suffix in ("fa4", "fp8", "flashqla"):
        condition = json.loads((EXPERIMENT / f"fp4_gdn_{suffix}.json").read_text())
        resolved = resolve_condition(condition, {})
        assert resolved["gdn_projection_precision"] == "fp4"
        assert resolved["quantization"] == "gleipnir_frost_gdn_fp4"
        assert resolved["baseline"] == "selected"
        assert resolved["high_reference"].endswith("fp4_gdn_projection02")


def test_mxfp8_compute_preserves_bf16_cache_and_requires_native_receipt():
    condition = json.loads((EXPERIMENT / "fp4_gdn_cudnn_mxfp8.json").read_text())
    resolved = resolve_condition(condition, {})
    assert "--kv-cache-dtype" not in resolved["extra_server_args"]
    assert resolved["gdn_projection_precision"] == "fp4"
    assert resolved["high_reference"].endswith("fp4_gdn_projection02")
    with pytest.raises(ValueError, match="validated forward-only"):
        resolve_condition({**condition, "mxfp8_validation": None}, {})
    with pytest.raises(ValueError, match="validated forward-only"):
        resolve_condition({**condition, "worker_cls": "Worker"}, {})
