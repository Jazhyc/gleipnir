"""Native FP4 selection is explicit and restricted to requested decoder layers."""

import pytest
import torch

from gleipnir import vllm_nvfp4


def test_mlp_selection_preserves_attention_vision_and_kept_layers(monkeypatch):
    monkeypatch.setenv("GLEIPNIR_NVFP4_SCOPE", "mlp")
    monkeypatch.setenv("GLEIPNIR_NVFP4_KEEP_LAYERS", "0,31")
    monkeypatch.setattr(vllm_nvfp4, "LinearBase", torch.nn.Linear)
    sentinel = object()
    monkeypatch.setattr(vllm_nvfp4, "NvFp4OnlineLinearMethod", lambda *args: sentinel)
    config = vllm_nvfp4.GleipnirNvFp4Config()
    layer = torch.nn.Linear(64, 32)
    assert (
        config.get_quant_method(layer, "model.layers.16.mlp.gate_up_proj") is sentinel
    )
    for prefix in (
        "model.layers.0.mlp.gate_up_proj",
        "model.layers.31.mlp.down_proj",
        "model.layers.16.self_attn.qkv_proj",
        "visual.blocks.0.mlp.fc1",
        "lm_head",
    ):
        assert isinstance(
            config.get_quant_method(layer, prefix), vllm_nvfp4.UnquantizedLinearMethod
        )
    assert config.get_quant_method(torch.nn.Identity(), "model.layers.16.mlp") is None


@pytest.mark.parametrize("field,value", [("SCOPE", "bogus"), ("BACKEND", "emulation")])
def test_unsupported_scope_or_fallback_rejected(monkeypatch, field, value):
    monkeypatch.setenv(f"GLEIPNIR_NVFP4_{field}", value)
    with pytest.raises(ValueError, match="Unsupported"):
        vllm_nvfp4.GleipnirNvFp4Config()


def test_cpu_weight_cannot_silently_use_emulation():
    with pytest.raises(ValueError, match="CUDA"):
        vllm_nvfp4.pack_weight(torch.zeros((32, 64), dtype=torch.bfloat16))
