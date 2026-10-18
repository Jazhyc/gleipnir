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


def mxfp8_case(tmp_path, **overrides):
    import hashlib
    import json

    from gleipnir.serving.quant_stack import mxfp8_receipt

    source = tmp_path / "kernel.py"
    source.write_text("kernel\n")
    receipt = {
        "precision": "mxfp8",
        "attention_shape": {"query_heads": 24, "kv_heads": 4, "head_dim": 256},
        "arithmetic_passed": True,
        "passed": False,
        "checks": [{"finite": True, "quantized_relative_l2": 0.004}] * 7,
        "base_equivalence": {"shape": [16, 4], "bitwise": [True, True]},
        "kv_metadata_contract": "TRTLLM cumulative pages plus exact token lengths",
        "gpu": "NVIDIA B200",
        "sources": {"kernel.py": hashlib.sha256(b"kernel\n").hexdigest()},
    } | overrides
    path = tmp_path / "mxfp8.json"
    path.write_text(json.dumps(receipt))
    binding = {
        "mxfp8_attention": {
            "validation": "mxfp8.json",
            "validation_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    }
    return lambda served: mxfp8_receipt(tmp_path, binding, served, "NVIDIA B200")


def test_mxfp8_receipt_binds_the_served_heads(tmp_path):
    check = mxfp8_case(tmp_path)
    assert check(text(64, 5120, 24, 4))["arithmetic_passed"]
    with pytest.raises(ValueError, match="does not cover"):
        check(text(32, 2560, 16, 4))


def test_mxfp8_receipt_requires_base_equivalence(tmp_path):
    check = mxfp8_case(tmp_path, base_equivalence={"bitwise": [True, False]})
    with pytest.raises(ValueError, match="does not cover"):
        check(text(64, 5120, 24, 4))


def test_mxfp8_receipt_rejects_source_drift(tmp_path):
    check = mxfp8_case(tmp_path)
    (tmp_path / "kernel.py").write_text("changed\n")
    with pytest.raises(ValueError, match="source drift"):
        check(text(64, 5120, 24, 4))


def qwen_text(hidden, mlp, value_heads):
    return SimpleNamespace(
        hidden_size=hidden,
        intermediate_size=mlp,
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_num_value_heads=value_heads,
        linear_value_head_dim=128,
    )


def test_fp4_tuning_shapes_reproduce_the_4b_table():
    from gleipnir.serving.fp4.tuning import SHAPES
    from gleipnir.serving.fp4_geometry_tuning import fp4_shapes

    assert fp4_shapes(qwen_text(2560, 9216, 32)) == SHAPES


def test_27b_fp4_shapes_and_producer_widths():
    from gleipnir.serving.fp4_geometry_producers import producer_k, vendor_k
    from gleipnir.serving.fp4_geometry_tuning import fp4_shapes

    text = qwen_text(5120, 17408, 48)
    assert fp4_shapes(text) == {
        "mlp_gate_up": (5120, 34816),
        "gdn_input": (5120, 16384),
        "mlp_down": (17408, 5120),
        "gdn_output": (6144, 5120),
    }
    assert producer_k(text) == {"silu": 17408, "norm": 5120}
    assert vendor_k(text) == {5120, 6144}
