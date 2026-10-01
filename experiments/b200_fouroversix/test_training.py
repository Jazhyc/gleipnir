from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.b200_fouroversix.run import make_job
from gleipnir.fouroversix_training import (
    FrozenFourOverSixLinear,
    FrozenFp4Runtime,
    install_eager_mlp_interfaces,
    install_mlp_precision,
    is_mlp_base,
    mlp_loading_skip_patterns,
)


@pytest.mark.parametrize("shape", [(17, 16), (1, 17, 16), (2, 17, 16), (8, 17, 16)])
def test_frozen_backward_matches_dense_for_arbitrary_batch(shape):
    weight = torch.randn(32, 16, dtype=torch.bfloat16)
    original = torch.nn.Linear(16, 32, bias=False, dtype=torch.bfloat16)
    original.weight = torch.nn.Parameter(weight, requires_grad=False)

    def matmul(inputs, packed, *, input_config):
        return inputs @ packed.T

    runtime = FrozenFp4Runtime(weight, weight.T, None, None, matmul)
    native = FrozenFourOverSixLinear(original, runtime)
    inputs = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_(True)
    grad = torch.randn((*shape[:-1], 32), dtype=torch.bfloat16)
    result = native(inputs)
    reference = torch.nn.functional.linear(reference_inputs, weight)
    result.backward(grad)
    reference.backward(grad)
    torch.testing.assert_close(result, reference)
    torch.testing.assert_close(inputs.grad, reference_inputs.grad)
    assert native.weight.grad is None
    assert runtime.forward_calls == runtime.backward_calls == 1


def test_only_mlp_bases_are_converted(monkeypatch):
    from gleipnir import fouroversix_training

    model = torch.nn.Module()
    model.mlp = torch.nn.Module()
    model.mlp.gate_proj = torch.nn.Module()
    model.mlp.gate_proj.base_layer = torch.nn.Linear(16, 32, bias=False)
    model.mlp.gate_proj.base_layer.requires_grad_(False)
    model.mlp.gate_proj.lora_A = torch.nn.Linear(16, 4, bias=False)
    model.self_attn = torch.nn.Linear(16, 16)
    untouched = model.self_attn.weight.detach().clone()
    monkeypatch.setattr(
        fouroversix_training, "native_runtime", lambda weight: SimpleNamespace()
    )
    metadata = install_mlp_precision(model, "fouroversix")
    assert isinstance(model.mlp.gate_proj.base_layer, FrozenFourOverSixLinear)
    assert model.mlp.gate_proj.base_layer.weight.dtype == torch.bfloat16
    assert model.mlp.gate_proj.lora_A.weight.requires_grad
    torch.testing.assert_close(model.self_attn.weight, untouched)
    assert metadata["converted_modules"] == ["mlp.gate_proj.base_layer"]
    assert not is_mlp_base("model.layers.0.self_attn.q_proj.base_layer")
    assert not is_mlp_base("model.layers.0.mlp.gate_proj.lora_A.default")


def test_trainable_base_fails_closed():
    with pytest.raises(ValueError, match="frozen"):
        FrozenFourOverSixLinear(torch.nn.Linear(16, 32, bias=False), SimpleNamespace())


def test_loading_exclusions_keep_original_mlp_and_attention_quantization():
    from transformers.quantizers.quantizers_utils import should_convert_module

    for precision in ["bf16", "fouroversix"]:
        patterns = mlp_loading_skip_patterns(precision)
        assert not should_convert_module("model.layers.0.mlp.gate_proj", patterns)
        assert not should_convert_module("model.layers.31.mlp.down_proj", patterns)
        assert not should_convert_module("lm_head", patterns)
        assert should_convert_module("model.layers.3.self_attn.q_proj", patterns)
        assert should_convert_module("model.layers.0.linear_attn.in_proj_qkv", patterns)
    assert mlp_loading_skip_patterns("nf4") is None


def test_eager_mlp_boundary_preserves_gradients_and_attention():
    model = torch.nn.Module()
    model.block = torch.nn.Module()
    model.block.mlp = torch.nn.Linear(16, 32)
    model.block.attention = torch.nn.Linear(16, 16)
    original_attention = model.block.attention.forward
    inputs = torch.randn(2, 7, 16, requires_grad=True)
    reference = model.block.mlp(inputs)
    assert install_eager_mlp_interfaces(model) == ["block.mlp"]
    actual = model.block.mlp(inputs)
    torch.testing.assert_close(actual, reference)
    actual.square().mean().backward()
    assert inputs.grad is not None and bool(torch.isfinite(inputs.grad).all())
    assert model.block.attention.forward == original_attention


def test_recipe_preserves_checkpointing_and_logical_batch():
    original = {
        "gradient_checkpointing_layer_indices": [0, 2, 5],
        "model_revision": "frozen",
        "rank": 128,
        "student_rows": "frozen.jsonl",
    }
    job = make_job(original, Path("/tmp/precision"), "fouroversix", 10)
    assert job["gradient_checkpointing_layer_indices"] == [0, 2, 5]
    assert job["rank"] == 128 and job["student_rows"] == "frozen.jsonl"
    assert job["micro_batch_size"] == 32 and job["gradient_accumulation_steps"] == 1
    assert job["adaptive_microbatching"] == {
        "enabled": True,
        "max_padded_tokens": 16384,
        "max_micro_batch_size": 8,
        "profile": False,
    }
