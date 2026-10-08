"""Audit the GDN intervention and restrict inherited FP4 audits to retained MLPs."""

import json
import os
from collections import Counter

import torch

from experiments.b200_attention_gdn_serving.worker import write
from experiments.b200_attention_precision.worker import AttentionPrecisionWorker
from experiments.b200_inference_benchmark.run import ROOT, sha
from gleipnir.serving.gdn_precision import (
    EXPECTED,
    SHAPES,
    projection_identity,
    validate_native,
)
from gleipnir.serving.vllm.gdn_precision import AuditedGdnFp8Method, install_audit


class GdnPrecisionWorker(AttentionPrecisionWorker):
    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        condition = self.vllm_config.additional_config["serving_condition"]
        path = condition["gdn_precision_validation"]
        receipt = json.loads((ROOT / path).read_text())
        validate_native(receipt)
        if receipt["gpu"] != torch.cuda.get_device_name():
            raise ValueError("GDN precision native hardware changed")
        for source, expected in receipt["sources"].items():
            if sha(ROOT / source) != expected:
                raise ValueError(f"GDN precision native source drift: {source}")
        super().load_model(load_dummy_weights=load_dummy_weights)
        linears = []
        for name, layer in self.model_runner.get_model().named_modules():
            identity = projection_identity(name)
            if identity is None:
                continue
            if identity not in EXPECTED or not isinstance(
                layer.quant_method, AuditedGdnFp8Method
            ):
                raise ValueError("GDN FP8 loaded scope changed")
            linears.append(
                {
                    "layer": name,
                    "identity": list(identity),
                    "dtype": str(layer.weight.dtype),
                    "weight_relative_l2": layer._gdn_fp8_weight_error,
                    "kernel": type(layer.quant_method.fp8_linear).__name__,
                }
            )
        if {tuple(r["identity"]) for r in linears} != EXPECTED:
            raise ValueError("GDN FP8 requires all 48 projections")
        calls, seen = [], set()
        audit = {
            "passed": False,
            "worker_pid": os.getpid(),
            "precision": "fp8",
            "validation_sha256": sha(ROOT / path),
            "linears": linears,
            "calls": calls,
        }

        def observed(name, rows):
            if name in seen:
                return
            identity = projection_identity(name)
            if identity not in EXPECTED or not 0 < rows <= 32768:
                raise ValueError("unexpected actual GDN FP8 dispatch")
            seen.add(name)
            calls.append(
                {"layer": name, "rows": rows, "shape": list(SHAPES[identity[1]])}
            )
            audit["passed"] = {projection_identity(n) for n in seen} == EXPECTED
            write("native_gdn_projections.json", audit)

        install_audit(observed)
        write("native_gdn_projections.json", audit)
        # The inherited audit expects 48 FP4 GDN producer calls. Those disappear
        # under FP8; retain its native receipts and audit only retained MLP paths.
        from gleipnir.serving.fp4 import integration, tuning
        from gleipnir.serving.vllm import frost_fp4 as frost

        preparation = {
            "passed": False,
            "worker_pid": os.getpid(),
            "calls": [],
            "expected_calls": {"norm": 32, "silu": 32},
            "scope": "retained FP4 MLP; no FP4 GDN producer",
        }

        def prepared(stage, shape):
            if torch.cuda.is_current_stream_capturing() or preparation["passed"]:
                return
            if stage not in {"norm", "silu"}:
                raise ValueError("unexpected FP4 preparation outside retained MLP")
            preparation["calls"].append({"stage": stage, "input_shape": list(shape)})
            if len(preparation["calls"]) == 64:
                preparation["passed"] = (
                    Counter(r["stage"] for r in preparation["calls"])
                    == preparation["expected_calls"]
                )
                if not preparation["passed"]:
                    raise ValueError("retained MLP producer scope changed")
            write("native_preparation.json", preparation)

        integration._AUDIT = prepared
        write("native_preparation.json", preparation)
        gemm = {
            "passed": False,
            "worker_pid": os.getpid(),
            "calls": [],
            "scope": "retained FP4 MLP tiles; GDN now native CUTLASS FP8",
        }
        tuned_seen = set()

        def tiled(name, rows, tile):
            if torch.cuda.is_current_stream_capturing():
                return
            key = name, tuning.row_band(rows), tile
            if key in tuned_seen:
                return
            tuned_seen.add(key)
            gemm["calls"].append(
                {"shape": name, "m": rows, "band": key[1], "tile": tile}
            )
            gemm["passed"] = {r["shape"] for r in gemm["calls"]} == {
                "mlp_gate_up",
                "mlp_down",
            }
            write("native_gemm_tuning.json", gemm)

        for name in ("mlp_gate_up", "mlp_down"):
            k, n = tuning.SHAPES[name]
            plan = frost._PLANS[(torch.cuda.current_device(), k, n)]
            if not isinstance(plan, tuning.ShapeTunedGemm):
                raise ValueError("retained FP4 MLP tile plan changed")
            previous = plan.audit

            def audited(rows, tile, name=name, previous=previous):
                previous(rows, tile)
                tiled(name, rows, tile)

            plan.audit = audited
        write("native_gemm_tuning.json", gemm)
        self.precision["gdn_fp8_actual_scope"] = {
            "projections": 48,
            "outputs": "BF16",
            "validation_sha256": sha(ROOT / path),
        }
        self.audit_serving_state()
        print("gdn_fp8_installed", len(linears), flush=True)
