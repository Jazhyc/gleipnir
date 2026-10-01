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
