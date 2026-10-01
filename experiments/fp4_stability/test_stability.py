"""Check the conservative backward contract independently of CUDA packing."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from experiments.fp4_stability.run import campaign_stages, validate_config
from gleipnir.fouroversix_training import (
    FrozenFourOverSixLinear,
    FrozenFp4Runtime,
    normalize_activation_rows,
)


def test_configs_are_bounded_and_invalid_campaign_fails_before_loading():
    root = Path(__file__).parent
    for name in [
        "config.yaml",
        "row_dequantized_diagnostic.yaml",
        "precision_cast_diagnostic.yaml",
    ]:
        config = yaml.safe_load((root / name).read_text())
        validate_config(config)
        assert config["diagnostics_only"]
        with pytest.raises(ValueError, match="global preflight"):
            validate_config({**config, "steps": 10})
        with pytest.raises(ValueError, match="ten matched"):
            validate_config({**config, "steps": 1000})
    training = yaml.safe_load((root / "row_dequantized_training.yaml").read_text())
    validate_config(training)
    assert not training["diagnostics_only"] and training["steps"] == 10
    matched_diagnostic = yaml.safe_load(
        (root / "row_precision_cast_diagnostic.yaml").read_text()
    )
    validate_config(matched_diagnostic)
    assert matched_diagnostic["diagnostics_only"]


def test_dequantized_backward_uses_forward_weight_not_master_or_transpose():
    torch.manual_seed(19)
    original = torch.nn.Linear(16, 32, bias=False, dtype=torch.bfloat16)
    original.requires_grad_(False)
    decoded = original.weight.detach().clone() + 0.125
    calls = []

    def native_matmul(inputs, weight, *, input_config):
        calls.append("native_forward")
        return inputs @ weight.T

    runtime = FrozenFp4Runtime(
        decoded,
        torch.zeros(16, 32, dtype=torch.bfloat16),
        None,
        None,
        native_matmul,
        dequantized_weight=decoded,
    )
    layer = FrozenFourOverSixLinear(original, runtime)
    inputs = torch.randn(2, 7, 16, dtype=torch.bfloat16, requires_grad=True)
    gradient = torch.randn(2, 7, 32, dtype=torch.bfloat16)
    layer(inputs).backward(gradient)
    torch.testing.assert_close(inputs.grad, gradient @ decoded)
    assert not torch.equal(inputs.grad, gradient @ original.weight)
    assert original.weight.grad is None
    assert calls == ["native_forward"]
    assert runtime.forward_calls == runtime.backward_calls == 1


def test_ten_update_campaign_requires_global_preflight_before_each_condition():
    source, longest = {"selection": "320"}, {"selection": "global32"}
    stages = campaign_stages(
        {"steps": 10, "conditions": ["fouroversix", "bf16"]}, source, longest
    )
    assert [(s["name"], s["steps"], s["source"]) for s in stages] == [
        ("fouroversix-global-preflight", 1, longest),
        ("fouroversix", 10, source),
        ("bf16-global-preflight", 1, longest),
        ("bf16", 10, source),
    ]
    assert campaign_stages(
        {"steps": 1, "conditions": ["fouroversix"]}, longest, longest
    ) == [dict(name="fouroversix", precision="fouroversix", source=longest, steps=1)]
    assert campaign_stages(
        {"steps": 10, "conditions": ["fouroversix"], "diagnostics_only": True},
        source,
        longest,
    ) == [dict(name="fouroversix", precision="fouroversix", source=source, steps=10)]


def test_row_scaling_is_independent_of_other_tokens_and_handles_zero():
    inputs = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [1.0, 2.0, -4.0, 0.0]], dtype=torch.bfloat16
    )
    normalized, scales = normalize_activation_rows(inputs)
    extended, extended_scales = normalize_activation_rows(
        torch.cat([inputs, torch.full((1, 4), 1e8, dtype=torch.bfloat16)])
    )
    torch.testing.assert_close(normalized, extended[:2], rtol=0, atol=0)
    torch.testing.assert_close(scales, extended_scales[:2], rtol=0, atol=0)
    torch.testing.assert_close(normalized.float() * scales, inputs.float())
    assert torch.isfinite(normalized).all()
    assert torch.equal(scales, torch.tensor([[1.0], [4.0]]))


@pytest.mark.parametrize("diagnostics_only", [False, True])
def test_failed_gate_never_updates_adapters(
    monkeypatch, tmp_path: Path, diagnostics_only
):
    from gleipnir import precision_training_screen as screen
    from gleipnir.adaptive_microbatching import MicrobatchPolicy

    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "cpu mock")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(total_memory=0),
    )
    monkeypatch.setattr(screen.importlib.metadata, "version", lambda name: "mock")
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList(
        [torch.nn.Linear(1, 1, bias=False) for _ in range(32)]
    )
    model.model = SimpleNamespace(layers=model.layers)
    with torch.no_grad():
        for layer in model.layers:
            layer.weight.fill_(1.0)

    def loss_forward(batch):
        value = torch.ones(1, 1)
        for layer in model.layers:
            value = layer(value)
        return value.sum()

    def install_compile():
        original = model.layers[0].forward
        model.layers[0].forward = lambda x: original(x) * 2
        return list(range(32))

    def forbidden(*args):
        pytest.fail("diagnostics must not create an optimizer or scheduler")

    arguments = dict(
        model=model,
        features=[{"direct_input_ids": list(range(16))} for _ in range(32)],
        collator=lambda items: items,
        loss_forward=loss_forward,
        optimizer_factory=forbidden,
        scheduler_factory=forbidden,
        install_compile=install_compile,
        output=tmp_path,
        seed=0,
        policy=MicrobatchPolicy(max_padded_tokens=16384, max_micro_batch_size=8),
        steps=1,
        max_grad_norm=1.0,
        metadata={},
        diagnostics_only=diagnostics_only,
    )
    if diagnostics_only:
        report = screen.run_precision_training_screen(**arguments)
        assert report["status"] == "diagnosed"
        assert report["prefix_compile_losses"][0]["loss"] == 1.0
        assert report["prefix_compile_losses"][1]["loss"] == 2.0
        assert report["master_unchanged"]
    else:
        with pytest.raises(ValueError, match="compilation loss canary failed"):
            screen.run_precision_training_screen(**arguments)
        report = json.loads((tmp_path / "screen.json").read_text())
        assert report["status"] == "failed"
        assert all(layer.weight.item() == 1.0 for layer in model.layers)
        assert all(layer.weight.grad is None for layer in model.layers)
    assert report["compile_canary"]["passed"] is False
    assert report["compile_canary"]["eager_repeat_loss"] == 1.0
    assert report["compile_canary"]["compiled_repeat_loss"] == 2.0
    assert report["steps"] == []
