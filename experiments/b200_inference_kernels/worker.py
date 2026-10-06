"""Audit stock online FP8 dispatch after the native vLLM loader runs."""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.online.fp8 import Fp8PtpcOnlineLinearMethod
from vllm.v1.worker.gpu_worker import Worker


class PrecisionAuditWorker(Worker):
    """Require 32 FP8 MLPs, BF16 elsewhere and an actual W8A8 GEMM method."""

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        super().load_model(load_dummy_weights=load_dummy_weights)
        audit, seen = {}, set()
        for name, layer in self.model_runner.get_model().named_modules():
            if not isinstance(layer, LinearBase):
                continue
            match = re.search(r"\.layers\.(\d+)\.mlp\.(gate_up_proj|down_proj)$", name)
            dtype = layer.weight.dtype
            method = layer.quant_method
            kernel = getattr(method, "fp8_linear", None)
            if match:
                if dtype != torch.float8_e4m3fn or not isinstance(
                    method, Fp8PtpcOnlineLinearMethod
                ):
                    raise ValueError(f"MLP is not stock W8A8 FP8: {name}")
                if kernel is None or "Marlin" in type(kernel).__name__:
                    raise ValueError(f"missing native W8A8 GEMM: {name}")
                seen.add((int(match[1]), match[2]))
            elif dtype != torch.bfloat16 or not isinstance(
                method, UnquantizedLinearMethod
            ):
                raise ValueError(f"unexpected non-MLP quantization: {name}")
            audit[name] = {
                "weight_dtype": str(dtype),
                "method": type(method).__name__,
                "kernel": type(kernel).__name__ if kernel else None,
                "shape": list(layer.weight.shape),
            }
        expected = {
            (index, projection)
            for index in range(32)
            for projection in ("gate_up_proj", "down_proj")
        }
        if seen != expected:
            raise ValueError("incomplete MLP precision coverage")
        root = Path(__file__).resolve().parents[2]
        path = root / "results/b200_inference_kernels/loaded_precision.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"passed": True, "mlp_projection_count": len(seen), "linears": audit},
                indent=2,
            )
            + "\n"
        )
        print(f"native_fp8_precision_audit_passed projections={len(seen)}", flush=True)
