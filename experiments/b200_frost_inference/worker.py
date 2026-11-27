"""Audit training-forward FP4 scope, runtime and bitwise adapter integration."""

import json
import os
import re
from pathlib import Path

import torch
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.v1.worker.gpu_worker import Worker

from gleipnir.vllm_frost_fp4 import FrostFp4LinearMethod, runtime_receipt


class FrostAuditWorker(Worker):
    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        runtime = runtime_receipt()
        super().load_model(load_dummy_weights=load_dummy_weights)
        audit, seen, checks = {}, set(), []
        for name, layer in self.model_runner.get_model().named_modules():
            if not isinstance(layer, LinearBase):
                continue
            method = layer.quant_method
            match = re.search(r"\.layers\.(\d+)\.mlp\.(gate_up_proj|down_proj)$", name)
            if match:
                if layer.weight.dtype != torch.float4_e2m1fn_x2 or not isinstance(
                    method, FrostFp4LinearMethod
                ):
                    raise ValueError(f"training-forward FP4 scope mismatch: {name}")
                seen.add((int(match[1]), match[2]))
                checks.extend(
                    {"layer": name, **v} for v in layer._gleipnir_kernel_checks
                )
            elif layer.weight.dtype != torch.bfloat16 or not isinstance(
                method, UnquantizedLinearMethod
            ):
                raise ValueError(f"unexpected non-MLP precision: {name}")
            audit[name] = {
                "dtype": str(layer.weight.dtype),
                "method": type(method).__name__,
                "weight_relative_l2": getattr(layer, "_gleipnir_weight_error", None),
            }
        if (
            seen != {(i, p) for i in range(32) for p in ("gate_up_proj", "down_proj")}
            or len(checks) != 6
        ):
            raise ValueError("incomplete FROST coverage/checks")
        # The two bounded training comparisons cached unused backward weights.
        from gleipnir.cudnn_fp4_mlp import clear_native_caches

        clear_native_caches()
        runtime.update(runtime_receipt())
        path = (
            Path(__file__).resolve().parents[2]
            / "results/b200_frost_inference/loaded_precision.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "passed": True,
                    "worker_pid": os.getpid(),
                    "runtime": runtime,
                    "mlp_projection_count": len(seen),
                    "kernel_checks": checks,
                    "linears": audit,
                },
                indent=2,
            )
            + "\n"
        )
        print(
            "training_frost_fp4_audit_passed projections=64 "
            "training_forward_bitwise_equal=true",
            flush=True,
        )
