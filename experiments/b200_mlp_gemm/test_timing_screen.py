"""Bounded timing authority preserves failed precision gates and isolation."""

from copy import deepcopy

import pytest

from gleipnir.packed_benchmark import summarize
from gleipnir.packed_training import validate_packed_training_config
from gleipnir.packed_training_screen import PackingCanaryError, _accept_canary


def student_config():
    return {
        "quantization": {
            "enabled": False,
            "full_bf16_lora": True,
            "mlp_precision": "bf16",
        },
        "model_loader": "causal_lm",
        "finetuning_mode": "lora",
        "attn_implementation": "sdpa",
        "lora": {"dropout": 0},
        "training": {
            "sequence_packing": True,
            "gated_delta_backend": "flashqla",
            "adaptive_microbatching": {"enabled": True},
            "selective_torch_compile_policy": "full_attention_and_linear_shell",
            "selective_torch_compile_canary_tokens": 256,
            "packed_attention_backend": "flash_attention_4",
            "packed_attention_version": "4.0.0b33",
            "packing_timing_authority": "User requests speed despite gradient drift",
            "native_fp4_mlp_timing": True,
            "max_steps": 20,
        },
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("native_fp4_mlp_timing", False),
        ("native_fp4_mlp_timing", "true"),
        ("max_steps", 21),
        ("max_steps", 0),
        ("startup_validation_reference", "old-receipt.json"),
        ("packing_timing_authority", ""),
    ],
)
def test_timing_config_rejects_unscoped_or_unbounded_requests(field, value):
    config = student_config()
    assert validate_packed_training_config(config)
    config["training"][field] = value
    with pytest.raises(ValueError, match="timing-only"):
        validate_packed_training_config(config)


def metadata():
    return {
        "optimizer_step_timing": {"durations_seconds": [4.0] * 20},
        "training_state": {"global_step": 20},
        "sequence_packing": {
            "initial_master_sha256": "initial",
            "final_master_sha256": "updated",
            "attention_backend": "flash_attention_4",
            "timing_authority": "User requests speed despite gradient drift",
            "preflight": {"passed": True},
            **{
                k: {
                    "passed": False,
                    "adapter_gradient_relative_l2": 0.679,
                    "accepted_for_learning_comparison": False,
                    "accepted_for_timing_comparison": True,
                }
                for k in ("eager_canary", "compiled_canary")
            },
        },
        "quantization": {
            "enabled": False,
            "full_bf16_lora": {
                "native_fp4_mlp": {
                    "base_forward": "nvfp4",
                    "base_input_gradient": "nvfp4",
                    "master_dtype": "float32",
                    "modules": [f"layers.{i}.mlp" for i in range(32)],
                }
            },
        },
        "checkpointed_layer_indices": [],
        "adaptive_microbatching": {"records": []},
        "train_metrics": {"train_runtime": 80.0},
        "peak_cuda_memory_allocated_bytes": 1,
        "selective_torch_compile": {},
    }


def test_timing_summary_requires_explicit_acceptance_and_keeps_failed_gates():
    m = metadata()
    with pytest.raises(ValueError, match="fresh packing gates"):
        summarize(m, 10, accept_learning=True)
    result = summarize(m, 10, accept_timing=True)
    assert result["measured_mean_seconds"] == 4.0
    assert result["packing_gates"]["eager_canary"]["passed"] is False
    assert (
        result["packing_gates"]["eager_canary"]["adapter_gradient_relative_l2"] == 0.679
    )


@pytest.mark.parametrize("missing", ["authority", "native", "backward", "memory"])
def test_timing_summary_rejects_missing_native_or_acceptance_evidence(missing):
    m = metadata()
    if missing == "authority":
        m["sequence_packing"]["timing_authority"] = None
    elif missing == "native":
        del m["quantization"]["full_bf16_lora"]["native_fp4_mlp"]
    elif missing == "backward":
        m["quantization"]["full_bf16_lora"]["native_fp4_mlp"][
            "base_input_gradient"
        ] = "bf16"
    else:
        m["sequence_packing"]["preflight"]["passed"] = False
    with pytest.raises(ValueError, match="fresh packing gates"):
        summarize(m, 10, accept_timing=True)


@pytest.mark.parametrize("invalid", ["leakage", "nonfinite"])
def test_timing_waiver_keeps_isolation_and_finite_gates(invalid):
    c = {
        "independent_loss": 1.11,
        "packed_loss": 1.06,
        "adapter_gradient_relative_l2": 0.679,
        "cases": [
            {
                "repeat_max_abs": 0.0,
                "perturb_max_abs": 0.0,
                "cross_input_grad_max_abs": 0.0,
                "own_input_grad_max_abs": 1.0,
            }
        ],
    }
    accepted = _accept_canary(deepcopy(c), 0.05, "explicit user timing request")
    assert accepted["passed"] is False
    assert accepted["accepted_for_learning_comparison"] is False
    assert accepted["accepted_for_timing_comparison"] is True
    if invalid == "leakage":
        c["cases"][0]["cross_input_grad_max_abs"] = 1.0
    else:
        c["adapter_gradient_relative_l2"] = float("nan")
    with pytest.raises(PackingCanaryError):
        _accept_canary(c, 0.05, "explicit user timing request")
