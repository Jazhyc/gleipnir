"""Install source-bound NVFP4 SwiGLU fusion on the tuned serving reference."""

import hashlib
import json
import os

import torch

from experiments.b200_attention_gdn_serving.prepare_worker import (
    PreparationMxfp8ServingAuditWorker,
)
from experiments.b200_attention_gdn_serving.worker import ROOT, write
from gleipnir.serving.sources import recorded_source_path


class SwigluPreparationMxfp8ServingAuditWorker(PreparationMxfp8ServingAuditWorker):
    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        from gleipnir.cudnn_fp4_epilogue import Nvfp4ScaledGemm
        from gleipnir.serving_fp4_swiglu import FusedSwiGlu, load_kernel
        from gleipnir.serving_fp4_swiglu_validation import SELECTED, validate_native
        from gleipnir.serving_fp4_tuning import SHAPES, ShapeTunedGemm, retile
        from gleipnir.serving_fp4_tuning_validation import validate_selection

        condition = self.vllm_config.additional_config["serving_condition"]
        path = condition["swiglu_fusion_validation"]
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
                raise ValueError(f"native SwiGLU source drift: {source}")
        if receipt["gpu"] != torch.cuda.get_device_name():
            raise ValueError("native SwiGLU hardware changed")
        tiled = json.loads((ROOT / condition["gemm_reference_validation"]).read_text())
        selected = validate_selection(tiled)
        for source, digest in tiled["sources"].items():
            if (
                hashlib.sha256(
                    recorded_source_path(ROOT, source).read_bytes()
                ).hexdigest()
                != digest
            ):
                raise ValueError(f"reference GEMM source drift: {source}")
        kernel, hashes = load_kernel(
            ROOT,
            ROOT / "results/b200_attention_gdn_serving/native_swiglu_source",
            n192_scale_fix=True,
        )
        if hashes != receipt["kernel"]:
            raise ValueError("native SwiGLU generated kernel drift")
        super().load_model(load_dummy_weights=load_dummy_weights)
        from gleipnir import vllm_frost_fp4 as frost
        from gleipnir.serving_fp4_swiglu_integration import install

        # Keep all selected reference projection plans, including the tiny-row
        # gate/up producer. Fused large rows have their own explicit receipt.
        for name, (k, n) in SHAPES.items():
            plans = {
                tile: retile(
                    Nvfp4ScaledGemm(frost.PLAN_ROWS, k, n, runtime_m=True), tile
                )
                for tile in set(selected[name].values())
            }
            frost._PLANS[(torch.cuda.current_device(), k, n)] = ShapeTunedGemm(
                plans, selected[name]
            )
        reference = next(
            iter(
                frost._PLANS[(torch.cuda.current_device(), 2560, 18432)].plans.values()
            )
        )
        plan = FusedSwiGlu(kernel, reference, (256, 192), (2, 1), vector=True)
        calls, seen = [], set()
        audit = {
            "passed": False,
            "worker_pid": os.getpid(),
            "validation_path": path,
            "validation_sha256": hashlib.sha256(raw).hexdigest(),
            "selected": SELECTED,
            "minimum_rows": 1536,
            "calls": calls,
            "condition": condition,
            **hashes,
            "validated_generated_sha256": receipt["kernel"]["generated_sha256"],
            "reference_gemm_selected": selected,
        }

        def observed(layer, m, backend):
            if torch.cuda.is_current_stream_capturing():
                return
            key = (layer, backend)
            if key in seen:
                return
            seen.add(key)
            calls.append(
                {
                    "layer": layer,
                    "rows": m,
                    "backend": backend,
                    "selected": SELECTED if backend == "fused" else "selected_frost",
                }
            )
            audit["passed"] = {
                c["layer"] for c in calls if c["backend"] == "fused"
            } == set(range(32))
            write("native_swiglu.json", audit)

        scope = install(self.model_runner.get_model(), plan, observed)
        self.precision["swiglu_fusion"] = {
            **scope,
            **hashes,
            "selected": SELECTED,
            "validation_sha256": audit["validation_sha256"],
        }
        self.audit_serving_state()
        write("native_swiglu.json", audit)
        print("nvfp4_gemm_swiglu_fusion_installed", scope, hashes, flush=True)
