"""Restore whole-row SwiGLU scaling and reject any direct-output execution."""

import json
import os

import torch

from experiments.b200_attention_gdn_serving.worker import write
from experiments.b200_attention_precision.worker import AttentionPrecisionWorker
from experiments.b200_inference_benchmark.run import ROOT, sha
from gleipnir.serving.fp4.swiglu_overhead_validation import validate_native


class WholeRowSwiGluWorker(AttentionPrecisionWorker):
    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        condition = self.vllm_config.additional_config["serving_condition"]
        if condition["swiglu_direct_fp4_output"] is not False:
            raise ValueError("ablation requires direct FP4 output disabled")
        path = condition["swiglu_output_ablation_validation"]
        receipt = json.loads((ROOT / path).read_text())
        validate_native(receipt)
        if receipt["gpu"] != torch.cuda.get_device_name():
            raise ValueError("whole-row SwiGLU hardware changed")
        if any(not r.get("changed_row_effect") for r in receipt["results"]):
            raise ValueError("whole-row SwiGLU replay lacks a changed-row effect")
        for source, expected in receipt["sources"].items():
            if sha(ROOT / source) != expected:
                raise ValueError(f"whole-row SwiGLU source drift: {source}")
        super().load_model(load_dummy_weights=load_dummy_weights)
        from gleipnir.serving.fp4 import (
            swiglu_native_output_integration as direct,
        )
        from gleipnir.serving.fp4 import (
            swiglu_overhead_integration as whole_row,
        )

        plan = direct._PLAN.overhead
        if plan.pad or plan.plan is None or plan.compile_count != 1:
            raise ValueError("existing symbolic whole-row plan is not ready")
        inherited = self.precision["swiglu_native_output"]
        inherited_audit = json.loads(
            (
                ROOT / "results/b200_attention_gdn_serving/native_swiglu_output.json"
            ).read_text()
        )
        if inherited["generated_sha256"] == receipt["kernel"]["generated_sha256"]:
            raise ValueError(
                "direct and whole-row kernel identities unexpectedly match"
            )
        if inherited_audit["overhead_reference_sha256"] != sha(
            ROOT / condition["swiglu_overhead_reference"]
        ):
            raise ValueError("inherited whole-row plan provenance changed")
        previous = json.loads(
            (ROOT / condition["swiglu_overhead_reference"]).read_text()
        )
        if previous["kernel"] != receipt["kernel"]:
            raise ValueError(
                "current native receipt validates a different whole-row kernel"
            )
        calls, seen = [], set()
        audit = {
            "passed": False,
            "worker_pid": os.getpid(),
            "validation_path": path,
            "validation_sha256": sha(ROOT / path),
            "calls": calls,
            "minimum_rows": 1536,
            "compile_count": plan.compile_count,
            **receipt["kernel"],
            "validated_generated_sha256": receipt["kernel"]["generated_sha256"],
            "direct_fp4_output_enabled": False,
            "direct_output_calls": 0,
            "activation_output": "BF16",
            "activation_packing": "whole-row dynamic FP4",
        }

        def observed(layer, rows, backend):
            if torch.cuda.is_current_stream_capturing() or (layer, backend) in seen:
                return
            if (
                layer not in range(32)
                or not 0 < rows <= 32768
                or backend not in {"fused", "reference_small_rows"}
            ):
                raise ValueError("unexpected whole-row SwiGLU dispatch")
            seen.add((layer, backend))
            calls.append({"layer": layer, "rows": rows, "backend": backend})
            audit["passed"] = {
                r["layer"] for r in calls if r["backend"] == "fused"
            } == set(range(32))
            write("native_swiglu_whole_row.json", audit)

        def forbidden(*args):
            audit["passed"] = False
            audit["direct_output_calls"] += 1
            write("native_swiglu_whole_row.json", audit)
            raise ValueError(
                "direct FP4 SwiGLU output executed in the disabled candidate"
            )

        direct._AUDIT = forbidden
        scope = whole_row.install(self.model_runner.get_model(), plan, observed)
        self.precision["swiglu_native_output"] = {
            "enabled": False,
            "direct_fp4_output_enabled": False,
            "inherited_validation_sha256": sha(
                ROOT / condition["swiglu_native_output_validation"]
            ),
            "replaced_by": "swiglu_whole_row",
        }
        self.precision["swiglu_whole_row"] = {
            **scope,
            **receipt["kernel"],
            "validation_sha256": sha(ROOT / path),
        }
        write("native_swiglu_whole_row.json", audit)
        write(
            "native_swiglu_output.json",
            {
                "passed": False,
                "enabled": False,
                "worker_pid": os.getpid(),
                "calls": [],
                "scope": "direct FP4 output disabled",
            },
        )
        self.audit_serving_state()
        print("swiglu_direct_fp4_output_disabled", scope, flush=True)
