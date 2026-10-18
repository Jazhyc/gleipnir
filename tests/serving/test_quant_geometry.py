from types import SimpleNamespace

import pytest

pytest.importorskip("vllm.model_executor.layers.quantization.online.fp8")

from gleipnir.serving.quant_stack import attention_weight_shape  # noqa: E402


def text(layers, hidden, heads, kv_heads):
    kinds = (["linear_attention"] * 3 + ["full_attention"]) * (layers // 4)
    return SimpleNamespace(
        layer_types=kinds,
        hidden_size=hidden,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=256,
        attn_output_gate=True,
    )


def test_4b_shapes_match_the_validated_recipe():
    config = text(32, 2560, 16, 4)
    assert attention_weight_shape(config, 3, "qkv_proj") == (10240, 2560)
    assert attention_weight_shape(config, 31, "o_proj") == (2560, 4096)


def test_27b_shapes_include_the_output_gate():
    config = text(64, 5120, 24, 4)
    assert attention_weight_shape(config, 63, "qkv_proj") == (14336, 5120)
    assert attention_weight_shape(config, 3, "o_proj") == (5120, 6144)


def test_linear_attention_layers_are_rejected():
    with pytest.raises(ValueError, match="not full attention"):
        attention_weight_shape(text(64, 5120, 24, 4), 0, "qkv_proj")
