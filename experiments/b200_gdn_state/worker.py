"""Audit native BF16 GDN state and every persistent recurrent-cache tensor."""

import hashlib
import json
import math
import os
from pathlib import Path

import torch

from experiments.b200_attention_gdn_serving.worker import ROOT, write
from experiments.b200_monitor_score.worker import MonitorScoreAuditWorker
from gleipnir.serving.gdn.state import make_forward, validate_native


class GdnStateMonitorScoreAuditWorker(MonitorScoreAuditWorker):
    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as gdn
        from flashinfer.gdn_prefill import chunk_gated_delta_rule

        path = (
            ROOT
            / self.vllm_config.additional_config["serving_condition"][
                "gdn_state_validation"
            ]
        )
        receipt = json.loads(path.read_text())
        validate_native(receipt)
        for source, expected in receipt["sources"].items():
            source_path = Path(source)
            if not source_path.is_absolute():
                source_path = ROOT / source_path
            if hashlib.sha256(source_path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"GDN state validation source drift: {source}")
        if self.vllm_config.cache_config.mamba_ssm_cache_dtype != "bfloat16":
            raise ValueError("BF16 state requires an explicit BF16 persistent cache")
        self._state_calls = []
        self._state_validation_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

        def observed(initial, final):
            if initial.dtype != torch.bfloat16 or (
                final is not None and final.dtype != torch.bfloat16
            ):
                raise ValueError("live GDN did not use BF16 state I/O")
            if (
                len(self._state_calls) < 8
                and not torch.cuda.is_current_stream_capturing()
            ):
                self._state_calls.append(
                    {
                        "input_shape": list(initial.shape),
                        "input_dtype": str(initial.dtype),
                        "output_dtype": str(final.dtype) if final is not None else None,
                    }
                )

        gdn.fi_chunk_gated_delta_rule = make_forward(
            chunk_gated_delta_rule, gdn.l2norm_fwd, observed=observed
        )
        super().load_model(load_dummy_weights=load_dummy_weights)
        layers = [
            m
            for m in self.model_runner.get_model().modules()
            if isinstance(m, gdn.ChunkGatedDeltaRule)
        ]
        if len(layers) != 24 or any(
            m.gdn_prefill_backend != "flashinfer" for m in layers
        ):
            raise ValueError("incomplete native BF16 GDN layer coverage")
        self.precision["gdn_state"] = {
            "state_dtype": "bfloat16",
            "gate_dtype": "float32",
            "accumulation_dtype": "float32",
            "prefill_layers": len(layers),
            "validation_sha256": self._state_validation_sha256,
        }
        self.audit_serving_state()
        print("bf16_gdn_state_load_passed layers=24", flush=True)

    def gdn_state_audit(self) -> dict:
        from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention

        caches = {}
        backing_storages = {}
        for name, layer in self.model_runner.get_model().named_modules():
            if isinstance(layer, GatedDeltaNetAttention):
                conv, state = layer.kv_cache
                if state.dtype != torch.bfloat16 or conv.dtype != torch.bfloat16:
                    raise ValueError(f"unexpected GDN cache precision: {name}")
                storage = state.untyped_storage()
                backing_storages[storage.data_ptr()] = storage.nbytes()
                caches[name] = {
                    "state_dtype": str(state.dtype),
                    "state_shape": list(state.shape),
                    "logical_state_bytes": state.numel() * state.element_size(),
                    "state_bytes_per_slot": math.prod(state.shape[1:])
                    * state.element_size(),
                    "conv_dtype": str(conv.dtype),
                }
        if len(caches) != 24 or not self._state_calls:
            raise ValueError("incomplete BF16 cache/native dispatch audit")
        receipt = {
            "passed": True,
            "worker_pid": os.getpid(),
            "caches": caches,
            "native_calls": self._state_calls,
            "logical_state_bytes": sum(
                c["logical_state_bytes"] for c in caches.values()
            ),
            "unique_backing_storage_bytes": sum(backing_storages.values()),
            "backing_storage_count": len(backing_storages),
            "memory_accounting": (
                "Logical views share hybrid cache backing; unique storage includes "
                "other cache views and is not dedicated recurrent-state memory."
            ),
            "gate_dtype": "float32",
            "accumulation_dtype": "float32",
            "validation_sha256": self._state_validation_sha256,
        }
        write("gdn_state.json", receipt)
        return receipt
