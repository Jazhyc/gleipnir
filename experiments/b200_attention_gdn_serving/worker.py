"""Bounded startup audit of FROST scope and native serving attention dispatch."""

import functools
import json
import os
from pathlib import Path
from typing import Any

import torch

from experiments.b200_frost_inference.worker import FrostAuditWorker

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/b200_attention_gdn_serving"


def write(name: str, receipt: dict) -> None:
    path = OUTPUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    temporary.replace(path)


class ServingAuditWorker(FrostAuditWorker):
    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        self.condition = self.vllm_config.additional_config["serving_condition"]
        self._runtime_audits_remaining = 2
        self._install_attention_audit()
        super().load_model(load_dummy_weights=load_dummy_weights)
        self.precision = json.loads(
            (ROOT / "results/b200_frost_inference/loaded_precision.json").read_text()
        )
        self.precision["serving_condition"] = self.condition
        self.audit_serving_state()

    def _install_attention_audit(self) -> None:
        import vllm.v1.attention.backends.flashinfer as backend

        original = backend.trtllm_batch_context_with_kv_cache
        calls = []
        expected = {
            "bf16": "torch.bfloat16",
            "fp8_e4m3": "torch.float8_e4m3fn",
        }[self.condition["attention_precision"]]
        write(
            "native_attention.json",
            {"passed": False, "worker_pid": os.getpid(), "calls": []},
        )

        @functools.wraps(original)
        def observed(*args, **kwargs):
            if torch.cuda.is_current_stream_capturing():
                return original(*args, **kwargs)
            query, cache = kwargs["query"], kwargs["kv_cache"]
            value = {
                "query_dtype": str(query.dtype),
                "cache_dtype": str(cache.dtype),
                "query_shape": list(query.shape),
                "cache_shape": list(cache.shape),
                "causal": kwargs.get("causal", True),
                "batch_size": int(kwargs["batch_size"]),
                "kernel": "flashinfer.prefill.trtllm_batch_context_with_kv_cache",
            }
            if (
                value["query_dtype"] != expected
                or value["cache_dtype"] != expected
                or query.shape[-2:] != (16, 256)
                or not value["causal"]
            ):
                raise ValueError(f"native attention dispatch mismatch: {value}")
            result = original(*args, **kwargs)
            calls.append(value)
            if len(calls) == 8:
                backend.trtllm_batch_context_with_kv_cache = original
                write(
                    "native_attention.json",
                    {
                        "passed": True,
                        "worker_pid": os.getpid(),
                        "serving_condition": self.condition,
                        "calls": calls,
                    },
                )
            return result

        backend.trtllm_batch_context_with_kv_cache = observed

    def audit_serving_state(self) -> None:
        from vllm.model_executor.layers.attention import Attention

        attention = {}
        for name, layer in self.model_runner.get_model().named_modules():
            if isinstance(layer, Attention):
                attention[name] = {
                    "implementation": type(layer.impl).__name__,
                    "kv_cache_dtype": layer.kv_cache_dtype,
                    "supports_quant_query_input": layer.impl.supports_quant_query_input,
                    "calculate_kv_scales": layer.calculate_kv_scales,
                    "q_scale": layer._q_scale_float,
                    "k_scale": layer._k_scale_float,
                    "v_scale": layer._v_scale_float,
                }
        if len(attention) != 8:
            raise ValueError("incomplete full-attention audit")
        self.precision["attention"] = attention
        write("loaded_precision.json", self.precision)

    def execute_model(self, scheduler_output: Any) -> Any:
        result = super().execute_model(scheduler_output)
        if self._runtime_audits_remaining:
            self.audit_serving_state()
            self._runtime_audits_remaining -= 1
        return result
