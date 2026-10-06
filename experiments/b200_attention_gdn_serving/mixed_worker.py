"""Audit mixed FROST MLP/GDN projection scope and native arithmetic."""

import os
import re

import torch
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.v1.worker.gpu_worker import Worker

from experiments.b200_attention_gdn_serving.worker import (
    ServingAuditWorker,
)
from gleipnir.cudnn_fp4_mlp import clear_native_caches
from gleipnir.serving_precision import is_gdn_projection
from gleipnir.vllm_frost_fp4 import FrostFp4LinearMethod, runtime_receipt
from gleipnir.vllm_frost_gdn import CheckedGdnFp8Method
from gleipnir.vllm_frost_gdn_fp4 import CheckedGdnFp4Method


class MixedServingAuditWorker(ServingAuditWorker):
    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        self.condition = self.vllm_config.additional_config["serving_condition"]
        precision = self.condition["gdn_projection_precision"]
        gdn_method, gdn_dtype = {
            "fp8": (CheckedGdnFp8Method, torch.float8_e4m3fn),
            "fp4": (CheckedGdnFp4Method, torch.float4_e2m1fn_x2),
        }[precision]
        self._runtime_audits_remaining = 2
        self._install_attention_audit()
        runtime_receipt()
        Worker.load_model(self, load_dummy_weights=load_dummy_weights)
        audit, mlps, gdns, mlp_checks, gdn_checks = {}, set(), set(), [], []
        for name, layer in self.model_runner.get_model().named_modules():
            if not isinstance(layer, LinearBase):
                continue
            method = layer.quant_method
            match = re.search(r"\.layers\.(\d+)\.mlp\.(gate_up_proj|down_proj)$", name)
            if match:
                if not isinstance(method, FrostFp4LinearMethod) or (
                    layer.weight.dtype != torch.float4_e2m1fn_x2
                ):
                    raise ValueError("FROST MLP scope changed")
                mlps.add((int(match[1]), match[2]))
                mlp_checks.extend(
                    {"layer": name, **check} for check in layer._gleipnir_kernel_checks
                )
            elif is_gdn_projection(name):
                if (
                    not isinstance(method, gdn_method)
                    or layer.weight.dtype != gdn_dtype
                ):
                    raise ValueError("GDN projection precision/scope changed")
                index = int(re.search(r"\.layers\.(\d+)\.", name)[1])
                gdns.add((index, name.rsplit(".", 1)[1]))
                gdn_checks.extend(
                    {"layer": name, **check}
                    for check in layer._gleipnir_gdn_kernel_checks
                )
            elif not isinstance(method, UnquantizedLinearMethod) or (
                layer.weight.dtype != torch.bfloat16
            ):
                raise ValueError(f"unexpected precision outside target scope: {name}")
            audit[name] = {
                "dtype": str(layer.weight.dtype),
                "method": type(method).__name__,
                "kernel": type(method.fp8_linear).__name__
                if isinstance(method, CheckedGdnFp8Method)
                else "FROST Native Nvfp4ScaledGemm"
                if isinstance(method, CheckedGdnFp4Method)
                else None,
            }
        if mlps != {(i, p) for i in range(32) for p in ("gate_up_proj", "down_proj")}:
            raise ValueError("incomplete FROST MLP coverage")
        expected = {
            (i, p)
            for i in range(32)
            if (i + 1) % 4 != 0
            for p in ("in_proj_qkvz", "out_proj")
        }
        if gdns != expected or len(mlp_checks) != 6 or len(gdn_checks) != 6:
            raise ValueError("incomplete GDN projection coverage/native checks")
        clear_native_caches()
        self.precision = {
            "passed": True,
            "worker_pid": os.getpid(),
            "runtime": runtime_receipt(),
            "serving_condition": self.condition,
            "mlp_projection_count": len(mlps),
            "gdn_projection_count": len(gdns),
            "kernel_checks": mlp_checks,
            "gdn_kernel_checks": gdn_checks,
            "linears": audit,
        }
        self.audit_serving_state()
        print("mixed_serving_audit_passed mlps=64 gdn_projections=48", flush=True)
