"""Audited symbolic-row SwiGLU on the selected attention-FP4 stack."""

import hashlib
import json
import os

import torch

from experiments.b200_attention_gdn_serving.attention_fp4_worker import (
    AttentionTunedPreparationMxfp8ServingAuditWorker,
)
from experiments.b200_attention_gdn_serving.worker import ROOT, write
from gleipnir.serving.sources import recorded_source_path


class OverheadAttentionTunedPreparationMxfp8ServingAuditWorker(
    AttentionTunedPreparationMxfp8ServingAuditWorker
):
    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        from gleipnir.serving_fp4_swiglu_overhead import OverheadSwiGlu, load_kernel
        from gleipnir.serving_fp4_swiglu_overhead_validation import validate_native

        condition = self.vllm_config.additional_config["serving_condition"]
        path = condition["swiglu_overhead_validation"]
        raw = (ROOT / path).read_bytes()
        receipt = json.loads(raw)
        validate_native(receipt)
        for source, digest in receipt["sources"].items():
            if (
                hashlib.sha256(
                    recorded_source_path(ROOT, source).read_bytes()
                ).hexdigest()
                != digest
            ):
                raise ValueError(f"symbolic SwiGLU source drift: {source}")
        if receipt["gpu"] != torch.cuda.get_device_name():
            raise ValueError("symbolic SwiGLU hardware changed")
        kernel, hashes = load_kernel(
            ROOT,
            ROOT / "results/b200_attention_gdn_serving/native_swiglu_overhead_source",
        )
        if hashes != receipt["kernel"]:
            raise ValueError("symbolic SwiGLU generated source drift")
        super().load_model(load_dummy_weights=load_dummy_weights)
        from gleipnir.serving_fp4_swiglu_overhead_integration import install

        plan = OverheadSwiGlu(kernel, pad=receipt["padding"])
        plan.compile()
        calls, seen = [], set()
        audit = {
            "passed": False,
            "worker_pid": os.getpid(),
            "validation_path": path,
            "validation_sha256": hashlib.sha256(raw).hexdigest(),
            "calls": calls,
            "minimum_rows": 1536,
            "compile_count": plan.compile_count,
            **hashes,
            "validated_generated_sha256": receipt["kernel"]["generated_sha256"],
        }

        def observed(layer, rows, backend):
            if torch.cuda.is_current_stream_capturing() or (layer, backend) in seen:
                return
            seen.add((layer, backend))
            calls.append({"layer": layer, "rows": rows, "backend": backend})
            audit["passed"] = {
                c["layer"] for c in calls if c["backend"] == "fused"
            } == set(range(32))
            write("native_swiglu_overhead.json", audit)

        scope = install(self.model_runner.get_model(), plan, observed)
        self.precision["swiglu_overhead"] = {**scope, **hashes}
        self.audit_serving_state()
        write("native_swiglu_overhead.json", audit)
        print("symbolic_swiglu_installed", scope, hashes, flush=True)
