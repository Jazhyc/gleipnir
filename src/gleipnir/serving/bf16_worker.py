"""Prove BF16 model/head weights and actual BF16 attention dispatch."""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.v1.worker.gpu_worker import Worker

from gleipnir.data.monitoring import write_json
from gleipnir.serving.precision import describe_attention_cache


class Bf16Worker(Worker):
    def save(self) -> None:
        write_json(Path(os.environ["GLEIPNIR_BF16_AUDIT"]), self.precision)
        write_json(
            Path(__file__).resolve().parents[3]
            / "results/b200_attention_gdn_serving/loaded_precision.json",
            self.precision,
        )

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        self.precision = {
            "passed": False,
            "worker_pid": os.getpid(),
            "attention_calls": [],
        }
        self.save()
        self.install_attention_audit()
        super().load_model(load_dummy_weights=load_dummy_weights)
        model = self.model_runner.get_model()
        config = self.vllm_config.model_config
        if config.quantization is not None or config.dtype != torch.bfloat16:
            raise ValueError("BF16 control cannot load a quantized model")
        if (
            config.runner_type != "pooling"
            or config.pooler_config.get_seq_pooling_type() != "LAST"
        ):
            raise ValueError("BF16 control requires causal LAST pooling")
        linears = {}
        for name, layer in model.named_modules():
            if isinstance(layer, LinearBase):
                if layer.weight.dtype != torch.bfloat16 or not isinstance(
                    layer.quant_method, UnquantizedLinearMethod
                ):
                    raise ValueError(f"non-BF16 projection: {name}")
                linears[name] = {
                    "dtype": str(layer.weight.dtype),
                    "method": type(layer.quant_method).__name__,
                }
        mlps = [n for n in linears if ".mlp." in n]
        gdns = [
            n
            for n in linears
            if n.endswith((".linear_attn.in_proj_qkvz", ".linear_attn.out_proj"))
        ]
        attn = [
            n
            for n in linears
            if n.endswith((".self_attn.qkv_proj", ".self_attn.o_proj"))
        ]
        if (len(mlps), len(gdns), len(attn)) != (64, 48, 16):
            raise ValueError(
                f"BF16 projection coverage drift: {(len(mlps), len(gdns), len(attn))}"
            )
        layers = [m for m in model.modules() if isinstance(m, Attention)]
        if len(layers) != 8 or any(m.kv_cache_dtype != "auto" for m in layers):
            raise ValueError("BF16 control requires eight unquantized attention caches")
        heads = {id(m.score): m.score for m in model.modules() if hasattr(m, "score")}
        if len(heads) != 1 or any(hasattr(m, "lm_head") for m in model.modules()):
            raise ValueError("BF16 control must retain exactly one two-row head")
        head = next(iter(heads.values()))
        if (
            head.weight.shape != (2, 2560)
            or head.weight.dtype != torch.bfloat16
            or head.bias is not None
        ):
            raise ValueError("BF16 classifier geometry changed")
        merged = Path(config.model)
        mapping = json.loads((merged / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        keys = [k for k in mapping if k.endswith("lm_head.weight")]
        if not keys and config.hf_text_config.tie_word_embeddings:
            keys = [k for k in mapping if k.endswith("embed_tokens.weight")]
        if len(keys) != 1:
            raise ValueError("ambiguous BF16 classifier source")
        with safe_open(merged / mapping[keys[0]], framework="pt") as f:
            expected = f.get_slice(keys[0])[15:17]
        if not torch.equal(head.weight.detach().cpu(), expected):
            raise ValueError("BF16 classifier source weight mismatch")
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            ChunkGatedDeltaRule,
            QwenGatedDeltaNetAttention,
        )

        recurrence = [m for m in model.modules() if isinstance(m, ChunkGatedDeltaRule)]
        gdn_layers = [
            m for m in model.modules() if isinstance(m, QwenGatedDeltaNetAttention)
        ]
        if (
            len(recurrence) != 24
            or any(m.gdn_prefill_backend != "flashinfer" for m in recurrence)
            or {m.gdn_decode_kernel for m in gdn_layers} != {"cuda"}
        ):
            raise ValueError("BF16 control changed the selected GDN backend")
        self.precision.update(
            passed=True,
            linears=linears,
            projection_counts={"mlp": 64, "gdn": 48, "attention": 16},
            head={"dtype": "torch.bfloat16", "shape": [2, 2560], "source": keys[0]},
            quantization=None,
        )
        self.save()
        print("bf16_projection_and_head_audit_passed", flush=True)

    def install_attention_audit(self) -> None:
        from vllm.v1.attention.backends import flashinfer

        original = flashinfer.trtllm_batch_context_with_kv_cache

        @functools.wraps(original)
        def observed(*args, **kwargs):
            if torch.cuda.is_current_stream_capturing():
                return original(*args, **kwargs)
            query = kwargs["query"]
            cache = describe_attention_cache(kwargs["kv_cache"])
            if (
                query.dtype != torch.bfloat16
                or cache["dtype"] != "torch.bfloat16"
                or not kwargs.get("causal", True)
            ):
                raise ValueError("BF16 native attention dtype/causality drift")
            value = original(*args, **kwargs)
            self.precision["attention_calls"].append(
                {
                    "query_dtype": str(query.dtype),
                    "cache_dtype": cache["dtype"],
                    "query_shape": list(query.shape),
                    "causal": True,
                }
            )
            if len(self.precision["attention_calls"]) == 8:
                self.save()
                flashinfer.trtllm_batch_context_with_kv_cache = original
            return value

        flashinfer.trtllm_batch_context_with_kv_cache = observed
