"""Check one-use packing identity, mutation invalidation and recomputation."""

from types import SimpleNamespace

import pytest
import torch
import yaml
from torch.utils.checkpoint import checkpoint

from experiments.fp4_stability.run import validate_config
from gleipnir.fouroversix_training import (
    FrozenFourOverSixLinear,
    FrozenFp4Runtime,
    PairedActivationCache,
    normalize_activation_rows,
)


@pytest.mark.parametrize("change", ["same", "clone", "mutation", "second_consumer"])
def test_cache_is_one_use_and_rejects_changed_inputs(change):
    cache = PairedActivationCache()
    values = torch.ones(3, 4)
    packed = object()
    assert cache.produce(values, lambda: (packed, None))[0] is packed
    if change == "clone":
        values = values.clone()
    elif change == "mutation":
        values.add_(1)
    elif change == "second_consumer":
        cache.consume(values, lambda: (None, None))
    fallback = object()
    actual = cache.consume(values, lambda: (fallback, None))[0]
    assert actual is (packed if change == "same" else fallback)
    assert cache.entry is None


def test_inference_tensor_without_version_uses_fresh_packing():
    cache = PairedActivationCache()
    with torch.inference_mode():
        values = torch.ones(2, 3)
        cache.produce(values, lambda: ("gate", None))
        assert cache.consume(values, lambda: ("up", None))[0] == "up"
    assert cache.entry is None and cache.misses == 1


@pytest.mark.parametrize("recompute", [False, True])
def test_shared_pair_preserves_outputs_gradients_and_releases_cache(
    monkeypatch, recompute
):
    import sys

    monkeypatch.setitem(
        sys.modules,
        "gleipnir.fp4_row_kernels",
        SimpleNamespace(
            normalize_rows=normalize_activation_rows,
            rescale_rows=lambda output, scales: (output.float() * scales).to(
                torch.bfloat16
            ),
        ),
    )
    cache = PairedActivationCache()
    layers = []
    for role in ["gate", "up"]:
        original = torch.nn.Linear(16, 32, bias=False, dtype=torch.bfloat16)
        original.requires_grad_(False)
        runtime = FrozenFp4Runtime(
            original.weight,
            None,
            None,
            None,
            lambda payload, weight, **kwargs: payload.decoded @ weight.T,
            dequantized_weight=original.weight,
            row_scaled_activations=True,
            fused_row_scaling=True,
            activation_cache=cache,
            activation_cache_role=role,
            activation_packer=lambda values, config: SimpleNamespace(decoded=values),
        )
        layers.append(FrozenFourOverSixLinear(original, runtime))
    values = torch.randn(2, 7, 16, dtype=torch.bfloat16, requires_grad=True)

    def action(inputs):
        return torch.nn.functional.silu(layers[0](inputs)) * layers[1](inputs)

    actual = (
        checkpoint(action, values, use_reentrant=False) if recompute else action(values)
    )
    reference_values = values.detach().clone().requires_grad_()
    normalized, scales = normalize_activation_rows(reference_values.reshape(-1, 16))
    reference_outputs = [
        ((normalized @ layer.weight.T).float() * scales).to(torch.bfloat16)
        for layer in layers
    ]
    reference = (
        torch.nn.functional.silu(reference_outputs[0]) * reference_outputs[1]
    ).reshape(2, 7, 32)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    # The base dX contract differentiates the decoded weight, not the row max.
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    assert bool(torch.isfinite(values.grad).all())
    assert cache.entry is None and cache.hits == (2 if recompute else 1)
    assert layers[0].runtime.activation_pack_calls == cache.hits
    assert layers[1].runtime.activation_pack_calls == 0
    assert all(layer.weight.grad is None for layer in layers)


def test_shared_campaign_fails_closed_for_precision_and_observation():
    from pathlib import Path

    config = yaml.safe_load(
        (Path(__file__).parent / "row_inductor_shared_timing.yaml").read_text()
    )
    validate_config(config)
    for overrides in [
        {"row_scaled_activations": False},
        {"fused_row_scaling": False},
        {"backward_mode": "fp4"},
        {"conditions": ["bf16"]},
        {"capture_native_operands": True},
    ]:
        with pytest.raises(ValueError, match="shared packing|per-token"):
            validate_config({**config, **overrides})
