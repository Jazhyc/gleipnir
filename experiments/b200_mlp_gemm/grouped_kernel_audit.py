"""Track native grouped TileLang specialization growth independently of timing."""

from __future__ import annotations

import importlib
from pathlib import Path

from transformers import TrainerCallback

from experiments.b200_mlp_gemm.resident_worker import write_json


class GroupedKernelAudit(TrainerCallback):
    """Check the four existing SM100 native JIT caches on each actual update."""

    def __init__(self, trial: Path):
        self.trial, self.records = trial, []

    @staticmethod
    def snapshot() -> dict[str, int]:
        functions = {
            "fused_fwd": "tilelang_fused_chunk_gdr_fwd",
            "fused_bwd": "tilelang_fused_chunk_gdr_bwd",
            "prepare_h": "tilelang_prepare_h",
            "kkt_solve": "tilelang_kkt_solve",
        }
        caches = {}
        for name, attribute in functions.items():
            module = importlib.import_module(
                f"flash_qla.ops.gated_delta_rule.chunk.blackwell.{name}"
            )
            cache = getattr(module, attribute)._kernel_cache
            caches[name] = len(cache)
        return caches

    def on_step_begin(self, args, state, control, **kwargs):
        self.before = self.snapshot()

    def on_step_end(self, args, state, control, **kwargs):
        after = self.snapshot()
        delta = {k: v - self.before[k] for k, v in after.items()}
        self.records.append({"step": state.global_step, "delta": delta})
        write_json(
            self.trial / "grouped_kernel_audit.json",
            {"updates": self.records, "specializations": after},
        )
        if state.global_step > 10 and any(delta.values()):
            raise ValueError(
                "measured grouped updates prepare TileLang specializations"
            )
