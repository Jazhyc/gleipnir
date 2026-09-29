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


@pytest.mark.parametrize(
    "field,value",
    [
        ("SCOPE", "bogus"),
        ("BACKEND", "emulation"),
        ("PACKER", "emulation"),
        ("SCALE_MODE", "unsafe"),
        ("PROJECTIONS", "bogus"),
        ("OTHER_MLP_PRECISION", "int4"),
    ],
)
def test_unsupported_scope_or_fallback_rejected(monkeypatch, field, value):
    monkeypatch.setenv(f"GLEIPNIR_NVFP4_{field}", value)
    with pytest.raises(ValueError, match="Unsupported"):
        vllm_nvfp4.GleipnirNvFp4Config()


def test_cpu_weight_cannot_silently_use_emulation():
    with pytest.raises(ValueError, match="CUDA"):
        vllm_nvfp4.pack_weight(torch.zeros((32, 64), dtype=torch.bfloat16))


def test_hybrid_selects_down_fp4_gate_up_fp8_and_attention_bf16(monkeypatch):
    monkeypatch.setenv("GLEIPNIR_NVFP4_SCOPE", "mlp")
    monkeypatch.setenv("GLEIPNIR_NVFP4_PROJECTIONS", "down")
    monkeypatch.setenv("GLEIPNIR_NVFP4_OTHER_MLP_PRECISION", "fp8_channel")
    monkeypatch.setenv("GLEIPNIR_NVFP4_KEEP_LAYERS", "0")
    monkeypatch.setattr(vllm_nvfp4, "LinearBase", torch.nn.Linear)
    fp4, fp8 = object(), object()
    monkeypatch.setattr(vllm_nvfp4, "NvFp4OnlineLinearMethod", lambda *args: fp4)
    monkeypatch.setattr(vllm_nvfp4, "Fp8PtpcOnlineLinearMethod", lambda: fp8)
    config = vllm_nvfp4.GleipnirNvFp4Config()
    layer = torch.nn.Linear(64, 32)
    assert config.get_quant_method(layer, "model.layers.16.mlp.down_proj") is fp4
    assert config.get_quant_method(layer, "model.layers.16.mlp.gate_up_proj") is fp8
    for prefix in (
        "model.layers.0.mlp.down_proj",
        "model.layers.16.self_attn.qkv_proj",
    ):
        assert isinstance(
            config.get_quant_method(layer, prefix), vllm_nvfp4.UnquantizedLinearMethod
        )


def test_fp8_layer_fallback_preserves_other_precision_paths(monkeypatch):
    monkeypatch.setenv("GLEIPNIR_NVFP4_SCOPE", "mlp")
    monkeypatch.setenv("GLEIPNIR_NVFP4_PROJECTIONS", "down")
    monkeypatch.setenv("GLEIPNIR_NVFP4_OTHER_MLP_PRECISION", "fp8_channel")
    monkeypatch.setenv("GLEIPNIR_NVFP4_FP8_LAYERS", "12,14")
    monkeypatch.setattr(vllm_nvfp4, "LinearBase", torch.nn.Linear)
    fp4, fp8 = object(), object()
    monkeypatch.setattr(vllm_nvfp4, "NvFp4OnlineLinearMethod", lambda *args: fp4)
    monkeypatch.setattr(vllm_nvfp4, "Fp8PtpcOnlineLinearMethod", lambda: fp8)
    config = vllm_nvfp4.GleipnirNvFp4Config()
    layer = torch.nn.Linear(64, 32)
    assert config.get_quant_method(layer, "model.layers.0.mlp.down_proj") is fp4
    assert config.get_quant_method(layer, "model.layers.12.mlp.down_proj") is fp8
    assert config.get_quant_method(layer, "model.layers.12.mlp.gate_up_proj") is fp8
    assert isinstance(
        config.get_quant_method(layer, "model.layers.12.self_attn.qkv_proj"),
        vllm_nvfp4.UnquantizedLinearMethod,
    )


def test_layer_fallback_audit_rejects_wrong_loaded_dtype():
    from types import SimpleNamespace

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(2)])
            for index, dtype in enumerate((torch.uint8, torch.float8_e4m3fn)):
                mlp = torch.nn.Module()
                mlp.down_proj = torch.nn.Module()
                mlp.down_proj.register_buffer(
                    "weight", torch.zeros((2, 2), dtype=dtype)
                )
                self.layers[index].mlp = mlp

    model = Model()
    config = SimpleNamespace(fp8_layers={1}, keep_layers=set())
    assert vllm_nvfp4.audit_layer_fallback(model, config) == {
        "0": "torch.uint8",
        "1": "torch.float8_e4m3fn",
    }
    config.fp8_layers = {0}
    with pytest.raises(ValueError, match="precision differs"):
        vllm_nvfp4.audit_layer_fallback(model, config)
