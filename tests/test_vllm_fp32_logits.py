"""FP32 logits retain the full head and explicitly request native output precision."""

import pytest
import torch

from gleipnir import vllm_nvfp4
from gleipnir.vllm_fp32_logits import Fp32LogitsMethod


def test_projection_requests_fp32_output_and_preserves_shape(monkeypatch):
    original_mm = torch.mm
    calls = []

    def mocked_mm(a, b, *, out_dtype):
        calls.append(out_dtype)
        return original_mm(a.float(), b.float())

    monkeypatch.setattr(torch, "mm", mocked_mm)
    layer = torch.nn.Linear(64, 32, bias=False).bfloat16()
    x = torch.ones(2, 3, 64, dtype=torch.bfloat16)
    result = Fp32LogitsMethod().apply(layer, x, torch.ones(32))
    assert result.shape == (2, 3, 32) and result.dtype == torch.float32
    assert calls == [torch.float32]
    with pytest.raises(ValueError, match="BF16/FP16"):
        Fp32LogitsMethod().apply(layer, x.float())


def test_none_scope_and_head_only_precision_are_explicit(monkeypatch):
    monkeypatch.setenv("GLEIPNIR_NVFP4_SCOPE", "none")
    monkeypatch.setenv("GLEIPNIR_NVFP4_FP32_LOGITS", "1")
    monkeypatch.setattr(vllm_nvfp4, "VocabParallelEmbedding", torch.nn.Embedding)
    monkeypatch.setattr(vllm_nvfp4, "LinearBase", torch.nn.Linear)
    config = vllm_nvfp4.GleipnirNvFp4Config()
    assert isinstance(
        config.get_quant_method(torch.nn.Embedding(32, 64), "lm_head"), Fp32LogitsMethod
    )
    assert isinstance(
        config.get_quant_method(
            torch.nn.Linear(64, 32), "model.layers.0.mlp.down_proj"
        ),
        vllm_nvfp4.UnquantizedLinearMethod,
    )


def test_tied_head_keeps_embedding_lookup(monkeypatch):
    monkeypatch.setenv("GLEIPNIR_NVFP4_SCOPE", "none")
    monkeypatch.setenv("GLEIPNIR_NVFP4_FP32_LOGITS", "1")
    monkeypatch.setattr(vllm_nvfp4, "VocabParallelEmbedding", torch.nn.Embedding)
    layer = torch.nn.Embedding(32, 64).bfloat16()
    method = vllm_nvfp4.GleipnirNvFp4Config().get_quant_method(
        layer, "model.embed_tokens"
    )
    assert isinstance(method, Fp32LogitsMethod)
    ids = torch.tensor([0, 1, 31])
    assert torch.equal(method.embedding(layer, ids), layer(ids))
