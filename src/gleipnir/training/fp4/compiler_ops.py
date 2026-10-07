"""Expose unchanged native FP4 arithmetic to the compiler as opaque operators.

Fake kernels describe only output metadata; real kernels call the existing
forward helper and decoded-BF16 dX arithmetic. This removes the Python graph
break at each frozen projection without tracing quantizer/TMA implementation.
"""

from __future__ import annotations

import weakref
from types import SimpleNamespace

import torch

from gleipnir.fouroversix_training import (
    FrozenFourOverSixLinear,
    FrozenFp4Function,
    FrozenFp4Runtime,
)

_RUNTIMES: weakref.WeakValueDictionary[int, FrozenFp4Runtime] = (
    weakref.WeakValueDictionary()
)


@torch.library.custom_op("gleipnir::frozen_fp4_forward", mutates_args=())
def frozen_fp4_forward(
    inputs: torch.Tensor,
    decoded_weight: torch.Tensor,
    runtime_key: torch.Tensor,
    out_features: int,
) -> torch.Tensor:
    """Run existing eager native kernels behind a compiler-visible boundary."""
    runtime = _RUNTIMES[int(runtime_key.item())]
    return FrozenFp4Function.forward(SimpleNamespace(), inputs, runtime)


@frozen_fp4_forward.register_fake
def _fake_forward(inputs, decoded_weight, runtime_key, out_features):
    return inputs.new_empty((*inputs.shape[:-1], out_features), dtype=torch.bfloat16)


@torch.library.custom_op("gleipnir::frozen_fp4_backward", mutates_args=())
def frozen_fp4_backward(
    gradient: torch.Tensor,
    decoded_weight: torch.Tensor,
    runtime_key: torch.Tensor,
    input_dtype: torch.dtype,
) -> torch.Tensor:
    """Retain the original BF16 decoded-weight input-gradient multiplication."""
    runtime = _RUNTIMES[int(runtime_key.item())]
    runtime.backward_calls += 1
    flat = gradient.reshape(-1, gradient.shape[-1]).to(torch.bfloat16)
    result = flat @ decoded_weight
    return result.reshape(*gradient.shape[:-1], decoded_weight.shape[-1]).to(
        input_dtype
    )


@frozen_fp4_backward.register_fake
def _fake_backward(gradient, decoded_weight, runtime_key, input_dtype):
    return gradient.new_empty(
        (*gradient.shape[:-1], decoded_weight.shape[-1]), dtype=input_dtype
    )


def _setup_context(ctx, inputs, output):
    activation, decoded_weight, runtime_key, _ = inputs
    ctx.save_for_backward(decoded_weight, runtime_key)
    ctx.input_dtype = activation.dtype


def _backward(ctx, gradient):
    decoded_weight, runtime_key = ctx.saved_tensors
    result = frozen_fp4_backward(
        gradient, decoded_weight, runtime_key, ctx.input_dtype
    )
    return result, None, None, None


frozen_fp4_forward.register_autograd(_backward, setup_context=_setup_context)


class CompilerVisibleFourOverSixLinear(FrozenFourOverSixLinear):
    """Preserve PEFT's frozen-linear interface without a Python graph break."""

    def __init__(self, original: torch.nn.Linear, runtime: FrozenFp4Runtime):
        if runtime.dequantized_weight is None or runtime.observer is not None:
            raise ValueError("compiler-visible FP4 requires unobserved BF16 backward")
        super().__init__(original, runtime)
        runtime_id = id(runtime)
        # An unregistered CPU tensor avoids one integer guard per projection.
        # Its lookup happens only inside the real opaque kernel, never on GPU.
        self.native_runtime_key = torch.tensor(
            runtime_id, dtype=torch.int64, device="cpu"
        )
        _RUNTIMES[runtime_id] = runtime

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return frozen_fp4_forward(
            inputs,
            self.runtime.dequantized_weight,
            self.native_runtime_key,
            self.out_features,
        )
