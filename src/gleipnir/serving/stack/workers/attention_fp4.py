"""Audit FP4 attention projections on the selected tuned reference."""

import hashlib
import json
import os

import torch

from gleipnir.serving.sources import recorded_source_matches
from gleipnir.serving.stack.workers.serving_audit import ROOT, write
from gleipnir.serving.stack.workers.tuned import (
    TunedPreparationMxfp8ServingAuditWorker,
)


class AttentionTunedPreparationMxfp8ServingAuditWorker(
    TunedPreparationMxfp8ServingAuditWorker
):
    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        from gleipnir.kernels.fp4.cudnn_fp4_epilogue import Nvfp4ScaledGemm
        from gleipnir.serving.fp4.attention import (
            EXPECTED,
            SHAPES,
            projection_identity,
            validate_native,
        )
        from gleipnir.serving.fp4.tuning import retile

        condition = self.vllm_config.additional_config["serving_condition"]
        if condition.get("attention_projection_precision") in {"bf16", "fp8"}:
            # Candidate-specific projection validation/auditing belongs to the
            # precision worker; preserve all inherited MLP/GDN/core checks.
            super().load_model(load_dummy_weights=load_dummy_weights)
            return
        path = condition["attention_projection_validation"]
        raw = (ROOT / path).read_bytes()
        receipt = json.loads(raw)
        validate_native(receipt)
        for source, digest in receipt["sources"].items():
            if not recorded_source_matches(ROOT, source, digest):
                raise ValueError(f"FP4 attention projection source drift: {source}")
        if receipt["gpu"] != torch.cuda.get_device_name():
            raise ValueError("FP4 projection hardware changed")
        super().load_model(load_dummy_weights=load_dummy_weights)
        from gleipnir import vllm_frost_fp4 as frost

        chosen = receipt["selected"]["qkv_proj"]
        if chosen["small"] != chosen["large"]:
            raise ValueError("unimplemented QKV tile split")
        k, n = SHAPES["qkv_proj"]
        frost._PLANS[(torch.cuda.current_device(), k, n)] = retile(
            Nvfp4ScaledGemm(frost.PLAN_ROWS, k, n, runtime_m=True), chosen["large"]
        )
        calls, seen = [], set()
        audit = {
            "passed": False,
            "worker_pid": os.getpid(),
            "validation_path": path,
            "validation_sha256": hashlib.sha256(raw).hexdigest(),
            "calls": calls,
            "condition": condition,
            "projection_count": 16,
            "projection_outputs": "BF16",
            "selected": receipt["selected"],
        }

        def observed(layer, rows, shape):
            if layer in seen:
                return
            if (
                projection_identity(layer) not in EXPECTED
                or tuple(shape) != SHAPES[projection_identity(layer)[1]]
            ):
                raise ValueError("unexpected native FP4 attention projection")
            seen.add(layer)
            calls.append({"layer": layer, "rows": rows, "shape": list(shape)})
            audit["passed"] = {
                projection_identity(c["layer"]) for c in calls
            } == EXPECTED
            write("native_attention_projections.json", audit)

        weights = {
            layer.weight.data_ptr(): name
            for name, layer in self.model_runner.get_model().named_modules()
            if projection_identity(name) in EXPECTED
        }

        class ObservedProjection:
            # Invoked inside the established opaque linear custom operator,
            # so scope logging never enters the Torch model graph.
            def __init__(self, plan):
                self.plan = plan

            def __call__(self, a, b):
                result = self.plan(a, b)
                if not torch.cuda.is_current_stream_capturing():
                    name = weights.get(b.codes.data_ptr())
                    if name is not None:
                        observed(
                            name, a.codes.shape[0], SHAPES[projection_identity(name)[1]]
                        )
                return result

        for k, n in SHAPES.values():
            key = (torch.cuda.current_device(), k, n)
            frost._PLANS[key] = ObservedProjection(frost._PLANS[key])
        self.precision["attention_projection_precision"] = {
            "dtype": "NVFP4",
            "projection_count": 16,
            "outputs": "BF16",
            "validation_sha256": audit["validation_sha256"],
        }
        self.audit_serving_state()
        write("native_attention_projections.json", audit)
        print(
            "attention_fp4_projection_installed",
            self.precision["attention_projection_precision"],
            flush=True,
        )
