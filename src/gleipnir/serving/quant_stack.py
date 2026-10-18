"""Geometry-derived FP8 attention / FP4 MLP-GDN serving for Qwen3.5-family models.

Phase 1 of the optimized recipe for models other than the validated 4B: the
unchanged FROST FP4 MLP and GDN methods, FP8 per-token/per-channel CUTLASS
full-attention projections whose expected shapes come from the served model's
config, BF16 FlashInfer attention, and the config-derived last-query and lean
GDN layers from ``bf16_stack``. The 4B default's hash-bound modules are reused
by import only. Not yet included: fused FP4 producers, native SwiGLU output,
tuned GEMM tiles and MXFP8 attention.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import register_quantization_config

from gleipnir.serving.bf16_stack import FastGdnMixin, LastQueryMixin
from gleipnir.serving.bf16_worker import Bf16Worker
from gleipnir.serving.fp4.attention import projection_identity
from gleipnir.serving.vllm import attention_precision
from gleipnir.serving.vllm.attention_precision import AttentionFp8Method, fp8_linear
from gleipnir.serving.vllm.frost_gdn_fp4 import CheckedGdnFp4Method, FrostGdnFp4Config

NAME = "gleipnir_geometry_fp8_fp4"


def attention_weight_shape(text: Any, layer: int, projection: str) -> tuple[int, int]:
    """Expected (out, in) weight shape of a full-attention projection."""
    if text.layer_types[layer] != "full_attention":
        raise ValueError(f"layer {layer} is not full attention")
    query = text.num_attention_heads * text.head_dim
    kv = text.num_key_value_heads * text.head_dim
    gate = 2 if getattr(text, "attn_output_gate", False) else 1
    if projection == "qkv_proj":
        return query * gate + 2 * kv, text.hidden_size
    if projection == "o_proj":
        return text.hidden_size, query
    raise ValueError(f"unknown attention projection {projection}")


class GeometryFp8Method(AttentionFp8Method):
    """FP8 attention projection checked against the model's own geometry."""

    def __init__(self, prefix: str, expected: tuple[int, int]) -> None:
        super().__init__(prefix)
        self.expected = expected

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        if tuple(layer.weight.shape) != self.expected:
            raise ValueError(
                f"unexpected attention projection {self.prefix}: "
                f"{tuple(layer.weight.shape)} != {self.expected}"
            )
        original = layer.weight
        if original.dtype != torch.bfloat16 or layer.bias is not None:
            raise ValueError("FP8 attention must quantize original merged BF16 weights")
        source = original[:16].float().clone()
        # Skip the 4B geometry check in the parent; keep its quantization.
        super(AttentionFp8Method, self).process_weights_after_loading(layer)
        if (
            type(self.fp8_linear).__name__ != "CutlassFP8ScaledMMLinearKernel"
            or layer.weight.dtype != torch.float8_e4m3fn
            or not bool(torch.isfinite(layer.weight_scale).all())
            or not bool((layer.weight_scale > 0).all())
        ):
            raise ValueError("FP8 attention requires native per-token/channel CUTLASS")
        decoded = layer.weight[:, :16].float().t() * layer.weight_scale[:16]
        layer._attention_fp8_weight_error = float(
            (decoded - source).norm() / source.norm().clamp_min(1e-12)
        )
        attention_precision._NAMES[layer.weight.data_ptr()] = self.prefix

    def apply(self, layer, x, bias=None):
        if bias is not None:
            raise ValueError("FP8 attention projection bias is unsupported")
        return fp8_linear(x, layer.weight, layer.weight_scale)


@register_quantization_config(NAME)
class GeometryFp8Fp4Config(FrostGdnFp4Config):
    @classmethod
    def get_name(cls) -> str:
        return NAME

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> GeometryFp8Fp4Config:
        return cls()

    def get_quant_method(self, layer, prefix):
        identity = projection_identity(prefix)
        if isinstance(layer, LinearBase) and identity is not None:
            text = get_current_vllm_config().model_config.hf_text_config
            return GeometryFp8Method(
                prefix, attention_weight_shape(text, identity[0], identity[1])
            )
        return super().get_quant_method(layer, prefix)


class QuantAuditWorker(Bf16Worker):
    """Audit FP8 attention, FP4 MLP and FP4 GDN coverage against the config.

    Writes the BF16 worker's audit schema (``passed``, ``attention_calls``,
    ``attention_layers``) plus projection counts, so the same harness checks it.
    """

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        from vllm.model_executor.layers.attention import Attention
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod
        from vllm.v1.worker.gpu_worker import Worker

        from gleipnir.serving.vllm.frost_fp4 import FrostFp4LinearMethod

        self.precision = {"passed": False, "worker_pid": os.getpid()}
        self.precision["attention_calls"] = []
        self.save()
        self.install_attention_audit()
        Worker.load_model(self, load_dummy_weights=load_dummy_weights)
        model = self.model_runner.get_model()
        config = self.vllm_config.model_config
        if config.quantization != NAME:
            raise ValueError(f"quantized audit requires {NAME}")
        kinds = list(config.hf_text_config.layer_types)
        full = kinds.count("full_attention")
        counts = {"attention_fp8": 0, "mlp_fp4": 0, "gdn_fp4": 0, "bf16": 0}
        for name, layer in model.named_modules():
            if not isinstance(layer, LinearBase):
                continue
            method = layer.quant_method
            if isinstance(method, GeometryFp8Method):
                if layer.weight.dtype != torch.float8_e4m3fn:
                    raise ValueError(f"FP8 attention not quantized: {name}")
                counts["attention_fp8"] += 1
            elif isinstance(method, CheckedGdnFp4Method):
                counts["gdn_fp4"] += 1
            elif isinstance(method, FrostFp4LinearMethod):
                counts["mlp_fp4"] += 1
            elif isinstance(method, UnquantizedLinearMethod):
                counts["bf16"] += 1
            else:
                raise ValueError(f"unexpected linear method {type(method)}: {name}")
        expected = {
            "attention_fp8": 2 * full,
            "mlp_fp4": 2 * len(kinds),
            "gdn_fp4": 2 * (len(kinds) - full),
        }
        if any(counts[k] != v for k, v in expected.items()):
            raise ValueError(f"quantized coverage drift: {counts} != {expected}")
        layers = [m for m in model.modules() if isinstance(m, Attention)]
        if len(layers) != full:
            raise ValueError("every full-attention layer must have an attention cache")
        self.attention_layers = full
        heads = {id(m.score): m.score for m in model.modules() if hasattr(m, "score")}
        if len(heads) != 1 or next(iter(heads.values())).weight.shape != (
            2,
            config.hf_text_config.hidden_size,
        ):
            raise ValueError("quantized monitor must keep one two-row head")
        self.precision.update(
            passed=True,
            quantization=NAME,
            projection_counts=counts,
            attention_layers=full,
        )
        self.save()
        print("quantized_projection_audit_passed", counts, flush=True)


class QuantLastQueryFastGdnWorker(FastGdnMixin, LastQueryMixin, QuantAuditWorker):
    """Phase 1 optimized stack: FP8/FP4 projections, last-query, lean GDN."""
