"""Projection selection preserves embeddings/head/vision and scopes other FP8."""

import pytest
import torch

from gleipnir import vllm_mixed_fp8 as mixed


@pytest.mark.parametrize("scope", ["all", "attention", "gdn"])
def test_mixed_selection_and_scope(monkeypatch, scope):
    monkeypatch.setenv("GLEIPNIR_MIXED_FP8_SCOPE", scope)
    monkeypatch.setattr(mixed, "LinearBase", torch.nn.Linear)
    channel, block = object(), object()
    monkeypatch.setattr(mixed, "Fp8PtpcOnlineLinearMethod", lambda: channel)
    monkeypatch.setattr(mixed, "TritonBlockFp8Method", lambda: block)
    config = mixed.GleipnirMixedFp8Config()
    layer = torch.nn.Linear(64, 32)
    assert config.get_quant_method(layer, "model.layers.0.mlp.gate_up_proj") is channel
    for name, target in [("self_attn", "attention"), ("linear_attn", "gdn")]:
        result = config.get_quant_method(layer, f"model.layers.0.{name}.in_proj")
        if scope in {"all", target}:
            assert result is block
        else:
            assert isinstance(result, mixed.UnquantizedLinearMethod)
    for prefix in ["lm_head", "visual.blocks.0.mlp.fc1"]:
        assert isinstance(
            config.get_quant_method(layer, prefix), mixed.UnquantizedLinearMethod
        )


def test_unknown_scope_fails_closed(monkeypatch):
    monkeypatch.setenv("GLEIPNIR_MIXED_FP8_SCOPE", "bogus")
    with pytest.raises(ValueError, match="scope"):
        mixed.GleipnirMixedFp8Config()


def test_forced_block_backend_cannot_silently_fall_back(monkeypatch):
    method = mixed.TritonBlockFp8Method.__new__(mixed.TritonBlockFp8Method)
    method.activation_quant_key, method.weight_quant_key = object(), object()
    method.input_dtype = method.out_dtype = torch.bfloat16
    monkeypatch.setattr(
        mixed.Fp8PerBlockOnlineLinearMethod, "create_weights", lambda *a, **kw: None
    )
    monkeypatch.setattr(mixed, "init_fp8_linear_kernel", lambda **kw: object())
    with pytest.raises(RuntimeError, match="Required Triton"):
        method.create_weights(torch.nn.Linear(64, 32))
