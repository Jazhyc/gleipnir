"""Audited direct FP4 SwiGLU output on the selected attention-FP4 stack."""

import hashlib
import json
import os

import torch

from experiments.b200_attention_gdn_serving.attention_fp4_worker import (
    AttentionTunedPreparationMxfp8ServingAuditWorker,
)
from experiments.b200_attention_gdn_serving.worker import ROOT, write


class NativeOutputAttentionTunedPreparationMxfp8ServingAuditWorker(
    AttentionTunedPreparationMxfp8ServingAuditWorker
):
    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        from gleipnir.serving_fp4_swiglu_native_output import (
            NativeOutputSwiGlu,
            load_kernel,
        )
        from gleipnir.serving_fp4_swiglu_native_output_validation import validate_native
        from gleipnir.serving_fp4_swiglu_overhead import OverheadSwiGlu
        from gleipnir.serving_fp4_swiglu_overhead import load_kernel as load_overhead
        from gleipnir.serving_fp4_swiglu_overhead_validation import (
            validate_native as validate_overhead,
        )

        condition = self.vllm_config.additional_config["serving_condition"]
        path = condition["swiglu_native_output_validation"]
        raw = (ROOT / path).read_bytes()
        receipt = json.loads(raw)
        validate_native(receipt)
        for source, digest in receipt["sources"].items():
            if hashlib.sha256((ROOT / source).read_bytes()).hexdigest() != digest:
                raise ValueError(f"symbolic SwiGLU source drift: {source}")
        if receipt["gpu"] != torch.cuda.get_device_name():
            raise ValueError("symbolic SwiGLU hardware changed")
        kernel, hashes = load_kernel(
            ROOT,
            ROOT / "results/b200_attention_gdn_serving/native_swiglu_output_source",
        )
        if hashes != receipt["kernel"]:
            raise ValueError("symbolic SwiGLU generated source drift")
        previous = json.loads(
            (ROOT / condition["swiglu_overhead_reference"]).read_text()
        )
        validate_overhead(previous)
        for source, digest in previous["sources"].items():
            if hashlib.sha256((ROOT / source).read_bytes()).hexdigest() != digest:
                raise ValueError(f"medium-row SwiGLU source drift: {source}")
        reference_kernel, reference_hashes = load_overhead(
            ROOT,
            ROOT / "results/b200_attention_gdn_serving/native_swiglu_medium_source",
        )
        if reference_hashes != previous["kernel"] or previous["gpu"] != receipt["gpu"]:
            raise ValueError("medium-row SwiGLU kernel/hardware drift")
        super().load_model(load_dummy_weights=load_dummy_weights)
        from gleipnir.serving_fp4_swiglu_native_output_integration import install

        plan = NativeOutputSwiGlu(kernel)
        plan.compile()
        plan.overhead = OverheadSwiGlu(reference_kernel, pad=False)
        plan.overhead.compile()
        calls, seen = [], set()
        audit = {
            "passed": False,
            "worker_pid": os.getpid(),
            "validation_path": path,
            "validation_sha256": hashlib.sha256(raw).hexdigest(),
            "calls": calls,
            "minimum_rows": 4097,
            "compile_count": plan.compile_count,
            "overhead_compile_count": plan.overhead.compile_count,
            "overhead_reference_sha256": hashlib.sha256(
                (ROOT / condition["swiglu_overhead_reference"]).read_bytes()
            ).hexdigest(),
            "scaling": receipt["scaling"],
            "strict_baseline_precision_passed": receipt[
                "strict_baseline_precision_passed"
            ],
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
            write("native_swiglu_output.json", audit)

        scope = install(self.model_runner.get_model(), plan, observed)
        scope.update(
            scaling=receipt["scaling"],
            global_inverse=1.0,
            packed_output="FP4 E2M1, E4M3 scales per 16 values",
            bf16_activation_stores=False,
            whole_row_scaling_preserved=False,
            activation_quantization_arithmetic_changed=True,
            native_output_minimum_rows=4097,
            medium_rows="1536..4096: validated whole-row-scaled overhead producer",
        )
        self.precision["swiglu_native_output"] = {**scope, **hashes}
        self.audit_serving_state()
        write("native_swiglu_output.json", audit)
        print("native_fp4_output_installed", scope, hashes, flush=True)
