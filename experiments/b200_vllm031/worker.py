"""Require every migrated GDN layer to use the explicitly requested backend."""

import os

from experiments.b200_attention_gdn_serving.worker import write
from experiments.b200_monitor_score.worker import MonitorScoreAuditWorker


class MigrationWorker(MonitorScoreAuditWorker):
    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        super().load_model(load_dummy_weights=load_dummy_weights)
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            ChunkGatedDeltaRule,
            QwenGatedDeltaNetAttention,
        )

        layers = [
            layer
            for layer in self.model_runner.get_model().modules()
            if isinstance(layer, ChunkGatedDeltaRule)
        ]
        if len(layers) != 24 or any(
            layer.gdn_prefill_backend != "flashinfer" for layer in layers
        ):
            raise ValueError("migration requires all 24 GDN layers on FlashInfer")
        attention = [
            layer
            for layer in self.model_runner.get_model().modules()
            if isinstance(layer, QwenGatedDeltaNetAttention)
        ]
        from flashinfer.gdn_prefill import chunk_gated_delta_rule

        nonparallel = getattr(chunk_gated_delta_rule, "_gleipnir_nonparallel", False)
        if nonparallel != (os.environ["GLEIPNIR_FLASHINFER_GDN_CP"] == "off"):
            raise ValueError("FlashInfer GDN context-parallel route was not installed")
        receipt = {
            "passed": True,
            "layers": 24,
            "backend": "flashinfer",
            "context_parallelism": "off" if nonparallel else "auto",
            "decode_kernels": sorted({layer.gdn_decode_kernel for layer in attention}),
            "fused_output_norm_layers": sum(
                layer.enable_fused_gdn_decode for layer in attention
            ),
        }
        self.precision["migration_gdn_backend"] = receipt
        self.audit_serving_state()
        write("migration_gdn_backend.json", receipt)
