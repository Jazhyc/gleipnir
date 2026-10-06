"""Capture selected FP4 model segments; leave attention and GDN eager."""

import hashlib
import inspect
import os
from collections import Counter
from pathlib import Path

from experiments.b200_attention_gdn_serving.swiglu_native_output_worker import (
    NativeOutputAttentionTunedPreparationMxfp8ServingAuditWorker,
)
from experiments.b200_attention_gdn_serving.worker import write
from gleipnir.serving_prefill_graphs import graph_padding_allowed, validate_graph_config


class PrefillGraphNativeOutputAttentionTunedPreparationMxfp8ServingAuditWorker(
    NativeOutputAttentionTunedPreparationMxfp8ServingAuditWorker
):
    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import BatchDescriptor

        condition = self.vllm_config.additional_config["serving_condition"]
        validate_graph_config(condition)
        dispatcher = self.model_runner.cudagraph_dispatcher
        source = Path(inspect.getfile(type(dispatcher)))
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != condition["prefill_graphs"]["dispatcher_sha256"]:
            raise ValueError("installed graph dispatcher changed")
        self._graph_enabled = True
        self._graph_live = False
        self._graph_decisions = Counter()
        self._graph_config = condition["compilation_config"]
        self._graph_source = {"path": str(source), "sha256": digest}
        original = dispatcher.dispatch
        limit = condition["prefill_graphs"]["max_padding_fraction"]

        def dispatch(num_tokens, *args, **kwargs):
            mode, desc = original(num_tokens, *args, **kwargs)
            reason = "upstream"
            if self._graph_live:
                if not self._graph_enabled:
                    mode, desc = CUDAGraphMode.NONE, BatchDescriptor(num_tokens)
                    reason = "canary_bypass"
                elif mode == CUDAGraphMode.PIECEWISE and not graph_padding_allowed(
                    num_tokens, desc.num_tokens, limit
                ):
                    mode, desc = CUDAGraphMode.NONE, BatchDescriptor(num_tokens)
                    reason = "padding_or_precision_band"
                self._graph_decisions[
                    (num_tokens, desc.num_tokens, mode.name, reason)
                ] += 1
            return mode, desc

        dispatcher.dispatch = dispatch
        super().load_model(load_dummy_weights=load_dummy_weights)
        self.precision["prefill_graphs"] = {
            "compilation_config": self._graph_config,
            "policy": condition["prefill_graphs"],
            "installed_dispatcher": self._graph_source,
            "attention_and_gdn": "eager splitting ops preserved",
        }
        self.audit_serving_state()
        self.prefill_graph_state()

    def execute_model(self, scheduler_output):
        self._graph_live = True
        try:
            return super().execute_model(scheduler_output)
        finally:
            self._graph_live = False

    def prefill_graph_state(self, enabled: str | None = None) -> dict:
        """Local benchmark RPC: toggle replay for a bounded numerical canary."""
        if enabled is not None:
            if enabled not in {"true", "false"}:
                raise ValueError("expected true or false")
            self._graph_enabled = enabled == "true"
        report = {
            "worker_pid": os.getpid(),
            "enabled": self._graph_enabled,
            "compilation_config": self._graph_config,
            "installed_dispatcher": self._graph_source,
            "decisions": [
                dict(rows=r, padded=p, mode=m, reason=why, count=n)
                for (r, p, m, why), n in self._graph_decisions.items()
            ],
            "dispatch_evidence_only": True,
            "replay_requires_profiler_confirmation": True,
        }
        write("prefill_graphs.json", report)
        return report
