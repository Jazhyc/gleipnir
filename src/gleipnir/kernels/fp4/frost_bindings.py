"""Scoped positional FROST dispatch for the validated FP4 training epilogue."""

from __future__ import annotations

import functools
import hashlib
import inspect
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

COMPILER_SHA256 = "3f0c4999f34b4797a812420b3f30341ab2f903284659861cfea316fa87bfc8b9"
_ROLE_NAMES = (
    "a_tensor",
    "b_tensor",
    "sfa_tensor",
    "sfb_tensor",
    "scale_tensor",
    "output_tensor",
)


def verify_compiler(source: Path) -> str:
    """Bind the lowered executor ABI before bypassing the public resolver."""
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != COMPILER_SHA256:
        raise ValueError("pinned lowered FROST compiler changed")
    return digest


class PositionalFrostJit:
    """Resolve graph tensor order once and retain the native lowered guards."""

    def __init__(self, plan: Any) -> None:
        self.original = plan.jit
        roles = tuple(getattr(plan, name) for name in _ROLE_NAMES)
        self.order = tuple(self.original.bound)
        if (
            len(self.order) != 6
            or len({id(t) for t in roles}) != 6
            or {id(t) for t in self.order} != {id(t) for t in roles}
        ):
            raise ValueError("direct bindings require the validated six-tensor graph")
        if plan.workspace_bytes or not callable(self.original.lowered):
            raise ValueError("direct bindings require a lowered zero-workspace plan")
        self.calls = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.original, name)

    def __call__(self, bindings: Mapping[Any, Any], *, stream: int) -> Any:
        if len(bindings) != 6:
            raise ValueError("direct bindings require exactly six operands")
        try:
            operands = [bindings[tensor] for tensor in self.order]
        except KeyError as error:
            raise ValueError(
                "direct bindings contain an unknown graph tensor"
            ) from error
        result = self.original.lowered(operands, stream=stream)
        self.calls += 1
        return result


class TrainingFrostBindings:
    """Switch drained training passes; original mode is the unwrapped control."""

    def __init__(self, plan_type: Any) -> None:
        self.plan_type = plan_type
        self.original_call = plan_type.__call__
        self.plans: dict[Any, PositionalFrostJit] = {}
        self.mode = "original"
        self.closed = False

        @functools.wraps(self.original_call)
        def execute(plan: Any, a: Any, b: Any) -> Any:
            if plan not in self.plans:
                self.plans[plan] = PositionalFrostJit(plan)
            proxy = self.plans[plan]
            if plan.jit is not proxy:
                if plan.jit is not proxy.original:
                    raise RuntimeError(
                        "FROST plan executor changed inside binding scope"
                    )
                plan.jit = proxy
            return self.original_call(plan, a, b)

        self.execute = execute

    def set_mode(self, mode: str) -> None:
        """Call only between completed passes, outside CUDA graph capture."""
        if self.closed:
            raise RuntimeError("FROST binding scope is closed")
        if mode not in {"original", "direct"}:
            raise ValueError("unknown FROST binding mode")
        if mode == "original":
            self.plan_type.__call__ = self.original_call
            drift = False
            for plan, proxy in self.plans.items():
                if plan.jit is not proxy and plan.jit is not proxy.original:
                    drift = True
                else:
                    plan.jit = proxy.original
            self.mode = mode
            if drift:
                raise RuntimeError("FROST plan executor changed inside binding scope")
        else:
            self.plan_type.__call__ = self.execute
        self.mode = mode

    def state(self) -> dict[str, Any]:
        """Return CPU-dispatch counts; graph replay does not increment them."""
        return {
            "mode": self.mode,
            "closed": self.closed,
            "plans": len(self.plans),
            "direct_calls": sum(proxy.calls for proxy in self.plans.values()),
            "geometries": [
                {"k": plan.plan.k, "n": plan.plan.n, "calls": proxy.calls}
                for plan, proxy in self.plans.items()
            ],
            "original_control_unwrapped": True,
            "arithmetic_changed": False,
            "activation_or_weight_values_cached": False,
        }

    def close(self) -> None:
        if not self.closed:
            try:
                self.set_mode("original")
            finally:
                self.closed = True


@contextmanager
def _binding_scope(plan_type: Any, mode: str) -> Iterator[TrainingFrostBindings]:
    """Internal injection seam for CPU tests of scope and restoration."""
    marker = "_gleipnir_training_frost_binding_owner"
    if hasattr(plan_type, marker):
        raise RuntimeError("FROST binding scope already installed")
    control = TrainingFrostBindings(plan_type)
    setattr(plan_type, marker, control)
    try:
        control.set_mode(mode)
        yield control
    finally:
        try:
            control.close()
        finally:
            delattr(plan_type, marker)


@contextmanager
def training_frost_bindings(
    mode: str = "direct",
) -> Iterator[TrainingFrostBindings]:
    """Opt into host-only dispatch, preserving the selected native recipe."""
    from cudnn.gemm.frost import compiler

    from gleipnir.kernels.fp4.cudnn_fp4_epilogue import Nvfp4ScaledGemm
    from gleipnir.training.backends.native_fp4 import validate_kernel_sources

    validate_kernel_sources()
    verify_compiler(Path(compiler.__file__))
    if inspect.getsourcefile(Nvfp4ScaledGemm.__call__) != inspect.getsourcefile(
        Nvfp4ScaledGemm
    ):
        raise RuntimeError("FROST plan call already overridden")
    with _binding_scope(Nvfp4ScaledGemm, mode) as control:
        yield control
