"""Source-bound long-history MXFP8 and measured CUDA memory telemetry."""

import json
import os
from pathlib import Path

import torch

from experiments.b200_long_context.envelope import enable, validate
from experiments.b200_monitor_score.worker import MonitorScoreAuditWorker

ROOT = Path(__file__).resolve().parents[2]


class LongContextWorker(MonitorScoreAuditWorker):
    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        import gleipnir.serving_mxfp8 as native

        self.envelope_path = ROOT / os.environ["GLEIPNIR_LONG_CONTEXT_VALIDATION"]
        receipt = json.loads(self.envelope_path.read_text())
        validate(receipt, ROOT)
        if receipt["gpu"] != torch.cuda.get_device_name():
            raise ValueError("extended envelope hardware changed")
        if (
            self.vllm_config.model_config.max_model_len != 262144
            or self.vllm_config.scheduler_config.max_num_batched_tokens != 32768
        ):
            raise ValueError("long-context launch envelope changed")
        self.history_shapes = set()
        enable(native)
        super().load_model(load_dummy_weights=load_dummy_weights)

    def _install_attention_audit(self) -> None:
        import vllm.v1.attention.backends.flashinfer as backend

        super()._install_attention_audit()
        original = backend.trtllm_batch_context_with_kv_cache

        def observed(*args, **kwargs):
            if not torch.cuda.is_current_stream_capturing():
                self.history_shapes.add(
                    (int(kwargs["max_q_len"]), int(kwargs["max_kv_len"]))
                )
            return original(*args, **kwargs)

        backend.trtllm_batch_context_with_kv_cache = observed

    def long_context_memory(self, *, reset: bool = False) -> dict:
        if reset:
            torch.cuda.reset_peak_memory_stats()
        return {
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "query_history_shapes": sorted(self.history_shapes),
        }
