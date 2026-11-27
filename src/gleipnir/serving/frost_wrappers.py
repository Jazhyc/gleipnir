"""Serving-only FROST binding reuse; retain NVIDIA's lowered guards and kernels."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

COMPILER_SHA = "3f0c4999f34b4797a812420b3f30341ab2f903284659861cfea316fa87bfc8b9"
_MODE = "direct"
_CALLS = {"original": 0, "direct": 0}
_WRAPPERS: dict[Any, DirectBindings] = {}
_ORIGINAL = None


def role_indices(plan: Any) -> tuple[int, ...]:
    """Resolve this fixed six-tensor graph once, preserving native operand order."""
    roles = [
        plan.a_tensor,
        plan.b_tensor,
        plan.sfa_tensor,
        plan.sfb_tensor,
        plan.scale_tensor,
        plan.output_tensor,
    ]
    bound = tuple(plan.jit.bound)
    positions = {id(tensor): index for index, tensor in enumerate(bound)}
    if len(bound) != 6 or set(positions) != {id(tensor) for tensor in roles}:
        raise ValueError("direct bindings require the validated six-tensor graph")
    if plan.workspace_bytes or plan.jit.lowered is None:
        raise ValueError("direct bindings require a lowered zero-workspace plan")
    return tuple(positions[id(tensor)] for tensor in roles)


def tensor_signature(tensor: Any) -> tuple:
    """Include storage/layout so replacement or relabelling cannot reuse stale views."""
    return (
        id(tensor),
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
    )


class WeightViews:
    """Bounded strong references to read-only weight views; never cache values."""

    def __init__(self, limit: int = 64) -> None:
        self.limit = limit
        self.entries: dict[tuple, tuple] = {}

    def get(self, codes: Any, scales: Any, factory: Any) -> tuple:
        key = tensor_signature(codes), tensor_signature(scales)
        if key not in self.entries:
            value = factory(codes, scales)
            if len(self.entries) >= self.limit:
                return value
            self.entries[key] = value
        return self.entries[key]


class DirectBindings:
    """Skip name/UID resolution, retaining fresh outputs and native runtime guards."""

    def __init__(self, plan: Any) -> None:
        self.plan = plan
        self.indices = role_indices(plan)
        self.weights = WeightViews()

    def __call__(self, a: Any, b: Any):
        import torch
        import triton
        from cudnn.gated_attention_block.kernels.proj_gemm import _rank3, _sf_view

        from gleipnir.cudnn_fp4_epilogue import _row_scale

        plan = self.plan
        m = plan._operand_rows(a, b)
        bw, bsf = self.weights.get(
            b.codes,
            b.scales,
            lambda codes, scales: (
                _rank3(codes, "b"),
                _sf_view(plan.plan, scales, "sf_b", plan.plan.n),
            ),
        )
        scale = torch.empty(m, device=a.codes.device, dtype=torch.float32)
        _row_scale[(triton.cdiv(m, 1024),)](a.inverse, b.inverse, scale, m, 1024)
        out = torch.empty(m, plan.plan.n, device=a.codes.device, dtype=torch.bfloat16)
        values = (
            _rank3(a.codes, "a"),
            bw,
            _sf_view(plan.plan, a.scales, "sf_a", m),
            bsf,
            scale.view(1, m, 1),
            out.unsqueeze(0),
        )
        operands = [None] * 6
        for index, value in zip(self.indices, values, strict=True):
            operands[index] = value
        plan.jit.lowered(
            operands, stream=torch.cuda.current_stream(out.device).cuda_stream
        )
        return out


def wrapper_state(mode: str | None = None) -> dict:
    """Control only host binding preparation between drained request passes."""
    global _MODE
    if mode is not None:
        if mode not in _CALLS:
            raise ValueError("unknown FROST host-wrapper mode")
        _MODE = mode
    return {
        "worker_pid": os.getpid(),
        "mode": _MODE,
        "calls": dict(_CALLS),
        "plans": len(_WRAPPERS),
        "cached_weights": sum(len(w.weights.entries) for w in _WRAPPERS.values()),
    }


def install_control(root: Path, validation: str) -> None:
    """Install inside the GPU worker; source-bind the optional host intervention."""
    global _ORIGINAL
    from cudnn.gemm.frost import compiler

    from gleipnir.cudnn_fp4_epilogue import Nvfp4ScaledGemm

    receipt = json.loads((root / validation).read_text())
    if (
        not receipt.get("passed")
        or receipt["helper_sha256"]
        != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    ):
        raise ValueError("host wrapper validation/source drift")
    if hashlib.sha256(Path(compiler.__file__).read_bytes()).hexdigest() != COMPILER_SHA:
        raise ValueError("pinned lowered FROST compiler changed")
    if _ORIGINAL is not None:
        raise RuntimeError("host wrapper control already installed")
    _ORIGINAL = Nvfp4ScaledGemm.__call__

    def execute(plan, a, b):
        if _MODE == "original":
            result = _ORIGINAL(plan, a, b)
        else:
            if plan not in _WRAPPERS:
                _WRAPPERS[plan] = DirectBindings(plan)
            result = _WRAPPERS[plan](a, b)
        _CALLS[_MODE] += 1
        return result

    Nvfp4ScaledGemm.__call__ = execute


def enable_worker_control(root: Path, validation: str) -> None:
    """Add opt-in initialization/RPC without changing the compiled worker class."""
    from experiments.b200_attention_gdn_serving.swiglu_native_output_worker import (
        NativeOutputAttentionTunedPreparationMxfp8ServingAuditWorker as Worker,
    )

    original_load = Worker.load_model

    def load(self, *, load_dummy_weights=False):
        install_control(root, validation)
        original_load(self, load_dummy_weights=load_dummy_weights)
        self.precision["frost_host_wrapper"] = {
            "validation": validation,
            "mode": _MODE,
            "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "native_lowered_guards_retained": True,
            "kernel_arithmetic_changed": False,
        }
        self.audit_serving_state()

    def state(self, mode=None):
        return wrapper_state(mode)

    Worker.load_model = load
    Worker.frost_wrapper_state = state
