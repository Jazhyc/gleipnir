"""Check opt-in activation selector routing and campaign rejection."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from experiments.fp4_stability.run import validate_config
from gleipnir.fouroversix_training import FrozenFp4Runtime, prepare_forward_activation


def test_alternative_selector_packs_once_without_a_shared_pair():
    calls = []

    def pack(values, config):
        payload = object()
        calls.append((values.clone(), config, payload))
        return payload

    runtime = FrozenFp4Runtime(
        None,
        None,
        "configuration",
        None,
        lambda *args: None,
        activation_packer=pack,
        activation_selector="fp16",
    )
    values = torch.randn(2, 7, 16)
    payload, scales = prepare_forward_activation(values, runtime)
    assert payload is calls[0][2] and scales is None
    torch.testing.assert_close(calls[0][0], values.reshape(-1, 16).to(torch.bfloat16))
    assert calls[0][1] == "configuration" and runtime.activation_pack_calls == 1


def test_selector_campaign_rejects_incompatible_modes():
    config = yaml.safe_load(
        (Path(__file__).parent / "row_inductor_fp16_selector_timing.yaml").read_text()
    )
    validate_config(config)
    for overrides in [
        {"activation_selector": "unknown"},
        {"row_scaled_activations": False},
        {"fused_row_scaling": False},
        {"fused_activation_packing": True},
        {"conditions": ["bf16"]},
        {"backward_mode": "fp4"},
        {"capture_native_operands": True},
    ]:
        with pytest.raises(ValueError, match="selector"):
            validate_config({**config, **overrides})


def test_strict_default_does_not_prepack_ordinary_projections():
    runtime = FrozenFp4Runtime(
        None, None, None, None, lambda *args: None, activation_packer=SimpleNamespace()
    )
    values = torch.randn(2, 7, 16, dtype=torch.bfloat16)
    actual, scales = prepare_forward_activation(values, runtime)
    assert isinstance(actual, torch.Tensor) and scales is None
    assert runtime.activation_pack_calls == 0
