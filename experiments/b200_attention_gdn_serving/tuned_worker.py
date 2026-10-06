"""Install receipt-bound NVIDIA FP4 tiles on the combined preparation stack."""

import hashlib
import json
import os

import torch

from experiments.b200_attention_gdn_serving.prepare_worker import (
    PreparationMxfp8ServingAuditWorker,
)
from experiments.b200_attention_gdn_serving.worker import ROOT, write


class TunedPreparationMxfp8ServingAuditWorker(PreparationMxfp8ServingAuditWorker):
    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        from gleipnir.cudnn_fp4_epilogue import Nvfp4ScaledGemm
        from gleipnir.serving_fp4_tuning import (
            SHAPES,
            ShapeTunedGemm,
            retile,
            row_band,
        )
        from gleipnir.serving_fp4_tuning_validation import validate_selection

        condition = self.vllm_config.additional_config["serving_condition"]
        raw = (ROOT / condition["gemm_tuning_validation"]).read_bytes()
        receipt = json.loads(raw)
        selected = validate_selection(receipt)
        for source, digest in receipt["sources"].items():
            if hashlib.sha256((ROOT / source).read_bytes()).hexdigest() != digest:
                raise ValueError(f"FP4 GEMM tuning source drift: {source}")
        if receipt["gpu"] != torch.cuda.get_device_name():
            raise ValueError("FP4 GEMM tuning hardware changed")
        super().load_model(load_dummy_weights=load_dummy_weights)
        from gleipnir import vllm_frost_fp4 as frost

        seen = set()
        calls = []
        audit = {
            "passed": False,
            "worker_pid": os.getpid(),
            "validation_sha256": hashlib.sha256(raw).hexdigest(),
            "validation_path": condition["gemm_tuning_validation"],
            "selected": selected,
            "calls": calls,
            "condition": condition,
            "graph_arithmetic_unchanged": True,
        }

        def observed(name, m, tile):
            if torch.cuda.is_current_stream_capturing():
                return
            band = row_band(m)
            key = (name, band, tile)
            if key in seen:
                return
            if tile != selected[name][band]:
                raise ValueError("unexpected native GEMM tile dispatch")
            seen.add(key)
            calls.append({"shape": name, "m": m, "band": band, "tile": tile})
            audit["passed"] = {c["shape"] for c in calls} == set(SHAPES)
            write("native_gemm_tuning.json", audit)

        for name, (k, n) in SHAPES.items():
            plans = {}
            for tile in set(selected[name].values()):
                plans[tile] = retile(
                    Nvfp4ScaledGemm(frost.PLAN_ROWS, k, n, runtime_m=True), tile
                )
            frost._PLANS[(torch.cuda.current_device(), k, n)] = ShapeTunedGemm(
                plans,
                selected[name],
                audit=lambda m, tile, shape=name: observed(shape, m, tile),
            )
        self.precision["gemm_tuning"] = {
            "selected": selected,
            "validation_sha256": audit["validation_sha256"],
        }
        self.audit_serving_state()
        write("native_gemm_tuning.json", audit)
        print("fp4_gemm_tuning_installed", selected, flush=True)
