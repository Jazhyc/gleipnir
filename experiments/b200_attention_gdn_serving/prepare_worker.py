"""Audit explicit FP4 preparation variants on the accepted MXFP8 stack."""

import hashlib
import json
import os
from collections import Counter

import torch

from experiments.b200_attention_gdn_serving.mxfp8_worker import Mxfp8ServingAuditWorker
from experiments.b200_attention_gdn_serving.worker import ROOT, write
from gleipnir.serving.sources import recorded_source_path


class PreparationMxfp8ServingAuditWorker(Mxfp8ServingAuditWorker):
    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        condition = self.vllm_config.additional_config["serving_condition"]
        mode = condition["fp4_preparation"]
        if self.vllm_config.parallel_config.tensor_parallel_size != 1:
            raise ValueError("FP4 producer fusion requires the pinned single-GPU model")
        paths = condition["fp4_prepare_validation"]
        if mode != "combined":
            paths = {mode: paths}
        elif set(paths) != {"vendor", "silu", "norm"}:
            raise ValueError("combined preparation requires all native receipts")
        receipts, hashes = {}, {}
        for stage, path in paths.items():
            raw = (ROOT / path).read_bytes()
            receipt = json.loads(raw)
            expected = {"vendor": "vendor_cuda", "silu": "silu", "norm": "norm"}[stage]
            if (
                not receipt.get("arithmetic_passed", receipt["passed"])
                or receipt["mode"] != expected
                or len(receipt["checks"]) != (12 if stage == "vendor" else 8)
                or any(
                    not r["finite"] or not r["gemm_relative_l2"] <= 0.01
                    for r in receipt["checks"]
                )
            ):
                raise ValueError("FP4 preparation native validation failed")
            if (
                not receipt["passed"]
                and not condition["allow_finite_parity_diagnostic"]
            ):
                raise ValueError(
                    "FP4 preparation strict precision failed; diagnostic disabled"
                )
            for source, expected_hash in receipt["sources"].items():
                if (
                    hashlib.sha256(
                        recorded_source_path(ROOT, source).read_bytes()
                    ).hexdigest()
                    != expected_hash
                ):
                    raise ValueError(f"FP4 preparation source drift: {source}")
            receipts[stage] = receipt
            hashes[stage] = hashlib.sha256(raw).hexdigest()
        # The unchanged baseline audits validate packed weights before installing
        # candidate activation arithmetic. Their receipts retain that scope.
        super().load_model(load_dummy_weights=load_dummy_weights)
        from gleipnir.serving_fp4_integration import install

        calls = []
        expected_calls = (
            {
                "vendor": 64
                if condition.get("attention_projection_precision") == "fp4"
                else 48,
                "silu": 32,
                "norm": 32,
            }
            if mode == "combined"
            else {mode: 112 if mode == "vendor" else 32}
        )
        count = sum(expected_calls.values())

        def observed(stage: str, shape: tuple) -> None:
            if len(calls) >= count or torch.cuda.is_current_stream_capturing():
                return
            calls.append({"stage": stage, "input_shape": list(shape)})
            if len(calls) == count:
                if Counter(c["stage"] for c in calls) != expected_calls:
                    raise ValueError("incomplete runtime preparation coverage")
                write(
                    "native_preparation.json",
                    {
                        "passed": True,
                        "worker_pid": os.getpid(),
                        "condition": condition,
                        "calls": calls,
                        "validation_sha256": hashes,
                        "weights_and_attention_unchanged": True,
                        "strict_preparation_precision_passed": all(
                            r["passed"] for r in receipts.values()
                        ),
                    },
                )

        warps = {
            stage: r["selected_warps"]
            for stage, r in receipts.items()
            if stage != "vendor"
        }
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
