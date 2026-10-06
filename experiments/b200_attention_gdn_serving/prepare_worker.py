"""Audit explicit FP4 preparation variants on the accepted MXFP8 stack."""

import hashlib
import json
import os

import torch

from experiments.b200_attention_gdn_serving.mxfp8_worker import Mxfp8ServingAuditWorker
from experiments.b200_attention_gdn_serving.worker import ROOT, write


class PreparationMxfp8ServingAuditWorker(Mxfp8ServingAuditWorker):
    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        condition = self.vllm_config.additional_config["serving_condition"]
        mode = condition["fp4_preparation"]
        if self.vllm_config.parallel_config.tensor_parallel_size != 1:
            raise ValueError("FP4 producer fusion requires the pinned single-GPU model")
        receipt = json.loads((ROOT / condition["fp4_prepare_validation"]).read_text())
        expected = {"vendor": "vendor_cuda", "silu": "silu", "norm": "norm"}[mode]
        if (
            not receipt.get("arithmetic_passed", receipt["passed"])
            or receipt["mode"] != expected
            or len(receipt["checks"]) != (12 if mode == "vendor" else 8)
            or any(
                not r["finite"] or r.get("gemm_relative_l2", 0) > 0.01
                for r in receipt["checks"]
            )
        ):
            raise ValueError("FP4 preparation native validation failed")
        if not receipt["passed"] and not condition["allow_finite_parity_diagnostic"]:
            raise ValueError(
                "FP4 preparation strict precision failed; diagnostic disabled"
            )
        for path, expected_hash in receipt["sources"].items():
            if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected_hash:
                raise ValueError(f"FP4 preparation source drift: {path}")
        # The unchanged baseline audits validate packed weights before installing
        # candidate activation arithmetic. Their receipts retain that scope.
        super().load_model(load_dummy_weights=load_dummy_weights)
        from gleipnir.serving_fp4_integration import install

        calls = []
        count = 112 if mode == "vendor" else 32

        def observed(stage: str, shape: tuple) -> None:
            if len(calls) >= count or torch.cuda.is_current_stream_capturing():
                return
            calls.append({"stage": stage, "input_shape": list(shape)})
            if len(calls) == count:
                write(
                    "native_preparation.json",
                    {
                        "passed": True,
                        "worker_pid": os.getpid(),
                        "condition": condition,
                        "calls": calls,
                        "validation_sha256": hashlib.sha256(
                            (ROOT / condition["fp4_prepare_validation"]).read_bytes()
                        ).hexdigest(),
                        "weights_and_attention_unchanged": True,
                        "strict_preparation_precision_passed": receipt["passed"],
                    },
                )

        warps = {mode: receipt["selected_warps"]} if mode != "vendor" else {}
        scope = install(
            self.model_runner.get_model(), mode, warps=warps, audit=observed
        )
        self.precision["activation_preparation"] = scope
        self.audit_serving_state()
        write(
            "native_preparation.json",
            {"passed": False, "worker_pid": os.getpid(), "scope": scope},
        )
        print("fp4_preparation_installed", mode, scope, flush=True)
