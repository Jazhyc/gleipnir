"""BF16 monitor workers with the arithmetic-preserving serving layers.

The FP8 default stacks last-query final-layer attention, the GDN route gate
and the lean GDN host path on its precision worker, with 4B layer counts in
its (hash-bound) worker sources. These mixins install the same shared kernels
and host paths on ``Bf16Worker``, deriving every count and index from the
served model's config, so they apply to any Qwen3.5-family dense model whose
final layer is full attention; anything else fails closed at load.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from experiments.b200_attention_gdn_serving.worker import write
from experiments.b200_gdn_route.worker import GdnRouteMixin
from gleipnir.serving.bf16_worker import Bf16Worker


def _layer_kinds(worker) -> list[str]:
    return list(worker.vllm_config.model_config.hf_text_config.layer_types)


class FastGdnMixin:
    """The lean GDN prefill host path on every linear-attention layer."""

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            QwenGatedDeltaNetAttention,
        )

        from gleipnir.serving import gdn_fast

        super().load_model(load_dummy_weights=load_dummy_weights)
        layers = [
            m
            for m in self.model_runner.get_model().modules()
            if isinstance(m, QwenGatedDeltaNetAttention)
        ]
        expected = sum(kind == "linear_attention" for kind in _layer_kinds(self))
        if not layers or len(layers) != expected:
            raise ValueError(f"GDN fast path expects {expected} GDN layers")
        runner = self.model_runner

        def fresh(num_prefills: int) -> bool:
            batch = runner.input_batch
            count = batch.num_reqs
            return (
                count == num_prefills
                and not batch.num_computed_tokens_cpu[:count].any()
            )

        gdn_fast.install(layers, fresh)
        write("gdn_fast.json", {"worker_pid": os.getpid(), "layers": len(layers)})

    def set_gdn_fast(
        self,
        enabled: bool,
        strided_norm: bool | None = None,
        fused_conv: bool | None = None,
    ) -> dict:
        from gleipnir.serving import gdn_fast

        return gdn_fast.set_enabled(enabled, strided_norm, fused_conv)

    def gdn_fast_audit(self) -> dict:
        from gleipnir.serving import gdn_fast

        return gdn_fast.audit()


class LastQueryMixin:
    """Last-query attention on the final layer, which must be full attention.

    Requires a BF16 bitwise receipt (``last_query_bf16.py``) for the served
    attention shape, bound through ``--additional-config``.
    """

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        import torch
        from vllm.forward_context import get_forward_context
        from vllm.model_executor.layers.attention import Attention
        from vllm.v1.attention.backends.flashinfer import TRTLLMPrefill

        from experiments.b200_last_layer_trim.worker import validate, validation_path
        from gleipnir.serving.last_query_attention import LastQueryState, install

        receipt = json.loads(
            validation_path(self.vllm_config.additional_config).read_text()
        )
        validate(receipt, torch.cuda.get_device_name())
        text = self.vllm_config.model_config.hf_text_config
        served = {
            "query_heads": text.num_attention_heads,
            "kv_heads": text.num_key_value_heads,
            "head_dim": text.head_dim,
        }
        shape = receipt.get("attention_shape", {})
        if receipt.get("precision") != "bf16" or any(
            shape.get(key) != value for key, value in served.items()
        ):
            raise ValueError(
                f"last-query receipt {shape} does not cover BF16 attention {served}"
            )
        super().load_model(load_dummy_weights=load_dummy_weights)
        pooler = self.vllm_config.model_config.pooler_config
        pooling = getattr(pooler, "seq_pooling_type", None) or getattr(
            pooler, "pooling_type", None
        )
        if pooling != "LAST" or not getattr(text, "is_causal", False):
            raise ValueError("last-query attention requires causal LAST pooling")
        kinds = _layer_kinds(self)
        layers = {}
        for name, module in self.model_runner.get_model().named_modules():
            match = re.search(r"layers\.(\d+)\.self_attn\.attn$", name)
            if match and isinstance(module, Attention):
                layers[int(match.group(1))] = (name, module)
        final = len(kinds) - 1
        if (
            kinds[final] != "full_attention"
            or max(layers, default=-1) != final
            or len(layers) != kinds.count("full_attention")
        ):
            raise ValueError("final decoder layer is not the last full attention")
        name, layer = layers[final]
        self.last_query = {
            "passed": False,
            "layer": name,
            "worker_pid": os.getpid(),
            "validation_sha256": receipt["sha256"],
            "calls": 0,
            "query_rows": 0,
            "attended_query_rows": 0,
            "prefills": 0,
            "attended_prefills": 0,
            "max_history_tokens": 0,
        }

        def observe(call: dict) -> None:
            audit = self.last_query
            audit["calls"] += 1
            audit["query_rows"] += call["tokens"]
            audit["attended_query_rows"] += (
                call["decode_tokens"] + call["attended_prefills"]
            )
            audit["prefills"] += call["prefills"]
            audit["attended_prefills"] += call["attended_prefills"]
            audit["max_history_tokens"] = max(
                audit["max_history_tokens"], call["max_history_tokens"]
            )
            if not audit["passed"]:
                audit["passed"] = True
                write("last_query_attention.json", audit)

        install(
            layer,
            prefill_type=TRTLLMPrefill,
            observe=observe,
            finished=None,
            state=LastQueryState(),
            current_context=get_forward_context,
        )
        self.precision["last_query_attention"] = {"layer": name}
        write("last_query_attention.json", self.last_query)
        print("last_query_attention_installed", name, flush=True)

    def last_query_audit(self) -> dict:
        return dict(self.last_query)


class Bf16FastGdnWorker(FastGdnMixin, Bf16Worker):
    """The lean GDN prefill host path on the BF16 worker."""


class Bf16LastQueryWorker(LastQueryMixin, Bf16Worker):
    """Last-query final-layer attention on the stock BF16 TRTLLM prefill."""


class Bf16LastQueryFastGdnWorker(FastGdnMixin, Bf16LastQueryWorker):
    """Last-query attention and the lean GDN host path on the BF16 worker."""


class Bf16GatedLastQueryFastGdnWorker(GdnRouteMixin, Bf16LastQueryFastGdnWorker):
    """The full BF16 stack: route gate, last-query attention, lean GDN path.

    The route threshold is model-dependent (FlashInfer's CP choice depends on
    heads x sequences); bind a threshold measured for the served model.
    """


def receipt_binding(root: Path, receipt: Path) -> dict:
    """``--additional-config`` entry that binds a BF16 last-query receipt."""
    from gleipnir.data.monitoring import file_hash

    return {
        "last_query_attention": {
            "validation": str(receipt.relative_to(root)),
            "validation_sha256": file_hash(receipt),
        }
    }
