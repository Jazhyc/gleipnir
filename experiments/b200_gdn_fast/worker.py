"""Adopted default worker with the lean GDN prefill host path installed."""

import os

from experiments.b200_attention_gdn_serving.worker import write
from experiments.b200_gdn_route.worker import GatedPrecisionWorker


class FastGdnWorker(GatedPrecisionWorker):
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
        if len(layers) != 24:
            raise ValueError("GDN fast path expects 24 GDN layers")
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

    def set_gdn_fast(self, enabled: bool) -> dict:
        from gleipnir.serving import gdn_fast

        return gdn_fast.set_enabled(enabled)

    def gdn_fast_audit(self) -> dict:
        from gleipnir.serving import gdn_fast

        return gdn_fast.audit()
