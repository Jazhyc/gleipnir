"""Register the online loader and prove native FP4 MLP coverage."""

import json
import os
import re
from pathlib import Path

import torch
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.v1.worker.gpu_worker import Worker

from gleipnir.vllm_online_nvfp4 import OnlineNvFp4Method


class Fp4AuditWorker(Worker):
    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        super().load_model(load_dummy_weights=load_dummy_weights)
        audit, seen, checks = {}, set(), []
        for name, layer in self.model_runner.get_model().named_modules():
            if not isinstance(layer, LinearBase):
                continue
            method = layer.quant_method
            match = re.search(r"\.layers\.(\d+)\.mlp\.(gate_up_proj|down_proj)$", name)
            if match:
                if layer.weight.dtype != torch.uint8 or not isinstance(
                    method, OnlineNvFp4Method
                ):
                    raise ValueError(f"MLP native FP4 scope mismatch: {name}")
                seen.add((int(match[1]), match[2]))
                checks.extend(
                    {"layer": name, **item}
                    for item in layer._gleipnir_fp4_kernel_checks
                )
            elif layer.weight.dtype != torch.bfloat16 or not isinstance(
                method, UnquantizedLinearMethod
            ):
                raise ValueError(f"non-MLP precision changed: {name}")
            audit[name] = {
                "dtype": str(layer.weight.dtype),
                "method": type(method).__name__,
                "kernel": type(method.kernel).__name__
                if isinstance(method, OnlineNvFp4Method)
                else None,
                "weight_relative_l2": getattr(
                    layer, "_gleipnir_fp4_weight_error", None
                ),
            }
        if (
            seen != {(i, p) for i in range(32) for p in ("gate_up_proj", "down_proj")}
            or len(checks) != 6
        ):
            raise ValueError("incomplete FP4 coverage or native arithmetic checks")
        path = (
            Path(__file__).resolve().parents[2]
            / "results/b200_fp4_inference/loaded_precision.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "passed": True,
                    "worker_pid": os.getpid(),
                    "mlp_projection_count": len(seen),
                    "kernel_checks": checks,
                    "linears": audit,
                },
                indent=2,
            )
            + "\n"
        )
        print(
            "native_cudnn_fp4_precision_and_arithmetic_passed projections=64",
            flush=True,
        )
