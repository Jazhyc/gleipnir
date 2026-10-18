"""Geometry-derived FP8 attention / FP4 MLP-GDN serving for Qwen3.5-family models.

Phase 1 of the optimized recipe for models other than the validated 4B: the
unchanged FROST FP4 MLP and GDN methods, FP8 per-token/per-channel CUTLASS
full-attention projections whose expected shapes come from the served model's
config, BF16 FlashInfer attention, and the config-derived last-query and lean
GDN layers from ``bf16_stack``. The 4B default's hash-bound modules are reused
by import only. Not yet included: fused FP4 producers, native SwiGLU output,
tuned GEMM tiles. ``Mxfp8AttentionMixin`` adds config-derived MXFP8 prefill
attention (phase 2).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import register_quantization_config

from gleipnir.data.monitoring import file_hash
from gleipnir.serving.bf16_stack import FastGdnMixin, LastQueryMixin
from gleipnir.serving.bf16_worker import Bf16Worker
from gleipnir.serving.fp4.attention import projection_identity
from gleipnir.serving.vllm import attention_precision
from gleipnir.serving.vllm.attention_precision import AttentionFp8Method, fp8_linear
from gleipnir.serving.vllm.frost_gdn_fp4 import CheckedGdnFp4Method, FrostGdnFp4Config

NAME = "gleipnir_geometry_fp8_fp4"
ROOT = Path(__file__).resolve().parents[3]


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


def mxfp8_receipt(root: Path, additional: dict, text: Any, gpu: str) -> dict:
    """Validate the bound ``mxfp8_canary`` receipt for the served attention."""
    binding = additional.get("mxfp8_attention") or {}
    path = root / binding.get("validation", "")
    if not path.is_file() or file_hash(path) != binding.get("validation_sha256"):
        raise ValueError("MXFP8 attention requires a hash-bound canary receipt")
    receipt = json.loads(path.read_text())
    served = {
        "query_heads": text.num_attention_heads,
        "kv_heads": text.num_key_value_heads,
        "head_dim": text.head_dim,
    }
    if (
        receipt.get("precision") != "mxfp8"
        or receipt.get("attention_shape") != served
        or not receipt.get("arithmetic_passed")
        or len(receipt.get("checks", [])) != 7
        or not all((receipt.get("base_equivalence") or {}).get("bitwise", [False]))
        or receipt.get("kv_metadata_contract")
        != "TRTLLM cumulative pages plus exact token lengths"
        or any(
            not c["finite"] or not c["quantized_relative_l2"] <= 0.01
            for c in receipt["checks"]
        )
    ):
        raise ValueError(f"MXFP8 receipt does not cover attention {served}")
    if receipt.get("gpu") != gpu:
        raise ValueError("MXFP8 receipt hardware changed")
    for source, expected in receipt["sources"].items():
        if file_hash(root / source) != expected:
            raise ValueError(f"MXFP8 validation source drift: {source}")
    return receipt


class Mxfp8AttentionMixin:
    """cuDNN MXFP8 prefill attention at the served model's GQA heads.

    Replaces vLLM's FlashInfer TRTLLM prefill call with
    ``mxfp8_geometry.paged_forward`` (decode stays BF16 FlashInfer) before the
    BF16 attention audit wraps it, so the audit observes, and then restores,
    the MXFP8 call. Last-query attention runs through the same call and needs
    an MXFP8 last-query receipt (``last_query_mxfp8.py``).
    """

    attention_precision = "mxfp8"

    def install_attention_audit(self) -> None:
        import torch
        import vllm.v1.attention.backends.flashinfer as backend

        from gleipnir.serving.mxfp8 import MAX_BATCH, MAX_LENGTH
        from gleipnir.serving.mxfp8_geometry import paged_forward

        config = self.vllm_config
        text = config.model_config.hf_text_config
        if (
            config.scheduler_config.max_num_seqs > MAX_BATCH
            or config.model_config.max_model_len > MAX_LENGTH
        ):
            raise ValueError("MXFP8 serving envelope is 128 sequences x 32K tokens")
        receipt = mxfp8_receipt(
            ROOT, config.additional_config, text, torch.cuda.get_device_name()
        )
        heads = {
            "query_heads": text.num_attention_heads,
            "kv_heads": text.num_key_value_heads,
        }
        self.precision["mxfp8_attention"] = {
            "passed": False,
            "validation_sha256": file_hash(
                ROOT / config.additional_config["mxfp8_attention"]["validation"]
            ),
            "strict_native_precision_passed": receipt["passed"],
            "kernel": "cuDNN SdpaFwdDslSm100 D256 MXFP8 THD",
            "decode": "unchanged BF16 FlashInfer",
            "calls": 0,
            **heads,
        }

        def forward(*args, **kwargs):
            if args:
                raise ValueError("MXFP8 serving expects the pinned keyword prefill API")
            result = paged_forward(**heads, **kwargs)
            audit = self.precision["mxfp8_attention"]
            if not torch.cuda.is_current_stream_capturing():
                audit["calls"] += 1
                if not audit["passed"]:
                    audit["passed"] = True
                    self.save()
            return result

        backend.trtllm_batch_context_with_kv_cache = forward
        super().install_attention_audit()


class TunedGemmMixin:
    """Install receipt-selected FROST FP4 tiles at the served model's shapes.

    The receipt (``experiments/qwen38_27b_serving/fp4_gemm_tune.py``) is bound
    through ``--additional-config`` ``fp4_gemm_tuning``; arithmetic is the same
    cuDNN graph as the untuned plan, recompiled with a pinned tile.
    """

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        import torch

        from gleipnir.kernels.fp4.cudnn_fp4_epilogue import Nvfp4ScaledGemm
        from gleipnir.serving.fp4.tuning import ShapeTunedGemm, retile, row_band
        from gleipnir.serving.fp4_geometry_tuning import fp4_shapes, validate_selection
        from gleipnir.serving.vllm import frost_fp4 as frost

        binding = self.vllm_config.additional_config.get("fp4_gemm_tuning") or {}
        path = ROOT / binding.get("validation", "")
        if not path.is_file() or file_hash(path) != binding.get("validation_sha256"):
            raise ValueError("FP4 GEMM tuning requires a hash-bound sweep receipt")
        receipt = json.loads(path.read_text())
        shapes = fp4_shapes(self.vllm_config.model_config.hf_text_config)
        selected = validate_selection(receipt, shapes)
        if receipt["gpu"] != torch.cuda.get_device_name():
            raise ValueError("FP4 GEMM tuning hardware changed")
        for source, digest in receipt["sources"].items():
            if file_hash(ROOT / source) != digest:
                raise ValueError(f"FP4 GEMM tuning source drift: {source}")
        super().load_model(load_dummy_weights=load_dummy_weights)
        audit = {
            "passed": False,
            "selected": selected,
            "validation_sha256": binding["validation_sha256"],
            "calls": [],
        }
        self.precision["gemm_tuning"] = audit
        seen = set()

        def observed(name, m, tile):
            if torch.cuda.is_current_stream_capturing():
                return
            key = (name, row_band(m))
            if key in seen:
                return
            if tile != selected[name][key[1]]:
                raise ValueError("unexpected native GEMM tile dispatch")
            seen.add(key)
            audit["calls"].append({"shape": name, "m": m, "band": key[1], "tile": tile})
            audit["passed"] = {c["shape"] for c in audit["calls"]} == set(shapes)
            self.save()

        for name, (k, n) in shapes.items():
            plans = {
                tile: retile(
                    Nvfp4ScaledGemm(frost.PLAN_ROWS, k, n, runtime_m=True), tile
                )
                for tile in set(selected[name].values())
            }
            frost._PLANS[frost.plan_key(torch.cuda.current_device(), k, n)] = (
                ShapeTunedGemm(
                    plans,
                    selected[name],
                    audit=lambda m, tile, shape=name: observed(shape, m, tile),
                )
            )
        self.save()
        print("fp4_gemm_tuning_installed", selected, flush=True)


class QuantLastQueryFastGdnWorker(FastGdnMixin, LastQueryMixin, QuantAuditWorker):
    """Phase 1 optimized stack: FP8/FP4 projections, last-query, lean GDN."""


class QuantMxfp8LastQueryFastGdnWorker(
    FastGdnMixin, LastQueryMixin, Mxfp8AttentionMixin, QuantAuditWorker
):
    """Phase 1 plus MXFP8 prefill attention (last-query runs on MXFP8)."""


class QuantTunedMxfp8LastQueryFastGdnWorker(
    TunedGemmMixin, QuantMxfp8LastQueryFastGdnWorker
):
    """MXFP8 attention stack with receipt-selected FP4 GEMM tiles."""
