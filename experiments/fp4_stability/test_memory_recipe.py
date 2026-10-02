"""Keep memory-policy experiments bounded and preserve the logical workload."""

from copy import deepcopy

import pytest
import torch

from experiments.fp4_stability.memory_recipe import (
    apply_memory_recipe,
    validate_memory_recipe,
)


def test_memory_recipe_changes_only_declared_execution_policy():
    source = {
        "selection_sha256": "frozen",
        "micro_batch_size": 32,
        "gradient_accumulation_steps": 1,
        "adaptive_microbatching": {
            "enabled": True,
            "max_padded_tokens": 16384,
            "max_micro_batch_size": 8,
        },
        "gradient_checkpointing_layer_indices": [0, 2, 5],
    }
    before = deepcopy(source)
    config = {
        "conditions": ["fouroversix"],
        "backward_mode": "dequantized_bf16",
        "reference_weights_on_cpu": True,
        "checkpoint_layer_indices": [0, 10, 21, 29],
        "adaptive_token_budget": 24576,
    }
    actual = apply_memory_recipe(source, config)
    assert source == before
    assert actual["selection_sha256"] == source["selection_sha256"]
    assert actual["micro_batch_size"] == 32
    assert actual["gradient_accumulation_steps"] == 1
    assert actual["adaptive_microbatching"]["max_micro_batch_size"] == 8
    assert actual["adaptive_microbatching"]["max_padded_tokens"] == 24576
    assert actual["gradient_checkpointing_layer_indices"] == [0, 10, 21, 29]


def test_no_checkpointing_bf16_preserves_workload_and_emits_training_override():
    from experiments.tool_trajectory_monitoring.run_distillation_train import (
        training_command,
    )

    source = {
        "gradient_checkpointing": True,
        "gradient_checkpointing_policy": "explicit",
        "gradient_checkpointing_layer_indices": [0, 2, 5],
        "selection_sha256": "frozen",
        "micro_batch_size": 32,
        "gradient_accumulation_steps": 1,
    }
    config = {
        "full_bf16_lora": True,
        "ten_step_learning_comparison": True,
        "conditions": ["bf16"],
        "steps": 10,
        "disable_gradient_checkpointing": True,
    }
    before = deepcopy(source)
    actual = apply_memory_recipe(source, config)
    assert source == before
    assert actual == dict(
        source,
        gradient_checkpointing=False,
        gradient_checkpointing_policy="all",
        gradient_checkpointing_layer_indices=None,
    )
    command = training_command(
        dict(
            actual,
            job_name="bf16",
            output_dir="results/test",
            seed=0,
            student_rows="data/rows.jsonl",
            soft_targets="data/targets.jsonl",
            causal_adapter_dir="results/test/adapter",
            max_length=29696,
            rank=128,
            lora_alpha=256,
            learning_rate=5e-5,
            num_train_epochs=-1,
            max_steps=10,
            save_steps=10,
        )
    )
    assert "student.training.gradient_checkpointing=false" in command
    assert "student.training.gradient_checkpointing_policy=all" in command
    assert not any("gradient_checkpointing_layer_indices=" in arg for arg in command)
    for override in [
        {"steps": 20},
        {"full_bf16_lora": False},
        {"disable_gradient_checkpointing": "true"},
        {"checkpoint_layer_indices": [0]},
    ]:
        with pytest.raises(ValueError):
            validate_memory_recipe(dict(config, **override))


@pytest.mark.parametrize(
    "overrides",
    [
        {"reference_weights_on_cpu": False},
        {"conditions": ["bf16"]},
        {"backward_mode": "fp4"},
        {"capture_native_operands": True},
        {"checkpoint_layer_indices": []},
        {"checkpoint_layer_indices": [2, 0]},
        {"checkpoint_layer_indices": [0, 0]},
        {"checkpoint_layer_indices": [32]},
        {"checkpoint_layer_indices": [True]},
        {"adaptive_token_budget": 65536},
    ],
)
def test_invalid_memory_policy_stops_before_model_loading(overrides):
    config = {
        "conditions": ["fouroversix"],
        "backward_mode": "dequantized_bf16",
        "reference_weights_on_cpu": True,
        "checkpoint_layer_indices": [0, 10, 21, 29],
    }
    with pytest.raises(ValueError):
        validate_memory_recipe({**config, **overrides})


def test_larger_bf16_physical_batch_preserves_checkpointing_and_logical_batch():
    source = {
        "gradient_checkpointing": True,
        "gradient_checkpointing_policy": "explicit",
        "gradient_checkpointing_layer_indices": [0, 2, 5],
        "micro_batch_size": 32,
        "gradient_accumulation_steps": 1,
        "selection_sha256": "frozen",
        "adaptive_microbatching": {
            "enabled": True,
            "max_padded_tokens": 16384,
            "max_micro_batch_size": 8,
        },
    }
    before = deepcopy(source)
    config = {
        "full_bf16_lora": True,
        "ten_step_learning_comparison": True,
        "conditions": ["bf16"],
        "steps": 10,
        "adaptive_token_budget": 32768,
    }
    actual = apply_memory_recipe(source, config)
    assert source == before
    assert actual == dict(
        source,
        adaptive_microbatching={
            "enabled": True,
            "max_padded_tokens": 32768,
            "max_micro_batch_size": 8,
        },
    )
    for override in [
        {"steps": 20},
        {"full_bf16_lora": False},
        {"adaptive_token_budget": 65536},
        {"disable_gradient_checkpointing": True},
    ]:
        with pytest.raises(ValueError):
            validate_memory_recipe(dict(config, **override))


def test_dense_reference_probe_restores_native_forward_after_failure():
    from gleipnir.fouroversix_training import FrozenFourOverSixLinear, FrozenFp4Runtime
    from gleipnir.precision_training_screen import dense_mlp_evaluation

    original = torch.nn.Linear(16, 32, bias=False, dtype=torch.bfloat16)
    original.requires_grad_(False)
    decoded = original.weight.detach().clone() + 0.125
    runtime = FrozenFp4Runtime(
        decoded,
        None,
        None,
        None,
        lambda inputs, weight, **kwargs: inputs @ weight.T,
        dequantized_weight=decoded,
    )
    layer = FrozenFourOverSixLinear(original, runtime)
    model = torch.nn.ModuleDict({"projection": layer})
    inputs = torch.randn(2, 16, dtype=torch.bfloat16)
    native = layer(inputs)
    with pytest.raises(RuntimeError, match="probe failed"):
        with dense_mlp_evaluation(model):
            torch.testing.assert_close(
                layer(inputs),
                torch.nn.functional.linear(inputs, original.weight),
                rtol=0,
                atol=0,
            )
            raise RuntimeError("probe failed")
    torch.testing.assert_close(layer(inputs), native, rtol=0, atol=0)
    assert layer.weight is original.weight and original.weight.grad is None


def test_reference_offload_rejects_missing_native_cache_without_mutation():
    from gleipnir.fouroversix_training import FrozenFourOverSixLinear, FrozenFp4Runtime
    from gleipnir.fp4_memory import offload_reference_weights

    original = torch.nn.Linear(16, 32, bias=False, dtype=torch.bfloat16)
    original.requires_grad_(False)
    runtime = FrozenFp4Runtime(None, None, None, None, lambda *args: None)
    layer = FrozenFourOverSixLinear(original, runtime)
    before = original.weight.detach().clone()
    with pytest.raises(ValueError, match="cached|native caches"):
        offload_reference_weights(layer)
    torch.testing.assert_close(original.weight, before, rtol=0, atol=0)
    assert original.weight.device.type == "cpu" and layer.weight is original.weight
