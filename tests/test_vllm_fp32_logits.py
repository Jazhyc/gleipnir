"""FP32 logits retain the full head and explicitly request native output precision."""

from types import SimpleNamespace

import pytest
import torch

from experiments.fp4_inference import worker
from gleipnir import vllm_fp32_logits, vllm_nvfp4
from gleipnir.vllm_fp32_logits import Fp32LogitsMethod, install_fp32_logits


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


def test_install_loaded_tied_head_and_reject_missing_head(monkeypatch):
    monkeypatch.setattr(vllm_fp32_logits, "VocabParallelEmbedding", torch.nn.Embedding)
    model = torch.nn.Module()
    model.embed_tokens = torch.nn.Embedding(32, 64).bfloat16()
    model.embed_tokens.quant_method = vllm_fp32_logits.UnquantizedEmbeddingMethod()
    model.lm_head = model.embed_tokens
    original_weight = model.lm_head.weight
    assert install_fp32_logits(model) == ["lm_head"]
    assert model.lm_head is model.embed_tokens
    assert model.lm_head.weight is original_weight
    assert isinstance(model.lm_head.quant_method, Fp32LogitsMethod)
    ids = torch.tensor([0, 1, 31])
    assert torch.equal(
        model.lm_head.quant_method.embedding(model.lm_head, ids),
        model.embed_tokens(ids),
    )
    with pytest.raises(ValueError, match="Expected one vocabulary head"):
        install_fp32_logits(torch.nn.Module())
    model.lm_head.quant_method = object()
    with pytest.raises(ValueError, match="unquantized"):
        install_fp32_logits(model)


def test_worker_uses_transferred_config_without_environment(monkeypatch):
    monkeypatch.delenv("GLEIPNIR_NVFP4_FP32_LOGITS", raising=False)
    monkeypatch.setattr(worker.Worker, "load_model", lambda self, **kwargs: None)
    calls = []
    monkeypatch.setattr(worker, "install_fp32_logits", calls.append)
    instance = worker.Fp4Worker.__new__(worker.Fp4Worker)
    instance.vllm_config = SimpleNamespace(
        quant_config=SimpleNamespace(fp32_logits="1")
    )
    model = object()
    instance.model_runner = SimpleNamespace(get_model=lambda: model)
    instance.load_model()
    assert calls == [model]
    instance.vllm_config.quant_config.fp32_logits = "0"
    instance.load_model()
    assert calls == [model]
