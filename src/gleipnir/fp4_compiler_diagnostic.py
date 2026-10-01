"""Observe native FP4 boundaries without adding compiler graph breaks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from gleipnir.fouroversix_training import FrozenFourOverSixLinear
from gleipnir.training_execution_audit import use_forwards


def tensor_difference(eager: torch.Tensor, compiled: torch.Tensor) -> dict[str, Any]:
    """Report value and layout differences without saving private activation tensors."""
    if eager.shape != compiled.shape:
        raise ValueError("observed eager/compiled operand shapes differ")
    difference = compiled.float() - eager.float()
    return {
        "shape": list(eager.shape),
        "eager_dtype": str(eager.dtype),
        "compiled_dtype": str(compiled.dtype),
        "relative_l2": float(difference.norm() / eager.float().norm().clamp_min(1e-12)),
        "maximum_absolute_difference": float(difference.abs().max()),
        "unequal_fraction": float((eager != compiled).float().mean()),
    }


def compare_native_operands(
    *,
    model: torch.nn.Module,
    batch: Any,
    loss_forward: Callable,
    original_forwards: list,
    eager_loss: float,
    compiled_loss: float,
) -> dict[str, Any]:
    """Compare first four layers at the existing opaque native projection boundary."""
    from fouroversix.quantize import dequantize, quantize_to_fp4
    from fouroversix.utils import QuantizeBackend

    selected = []
    for name, module in model.named_modules():
        if isinstance(module, FrozenFourOverSixLinear):
            parts = name.split(".")
            if int(parts[parts.index("layers") + 1]) < 4:
                selected.append((name, module))
    if not selected:
        raise ValueError("operand capture requires native Qwen decoder MLPs")
    captures = {"eager": {}, "compiled": {}}
    phase = "eager"
    observers = [(module.runtime, module.runtime.observer) for _, module in selected]
    for name, module in selected:
        runtime = module.runtime

        def observe(inputs, normalized, scales, output, name=name, runtime=runtime):
            packed = quantize_to_fp4(normalized, runtime.activation_config)
            decoded = dequantize(
                packed,
                backend=QuantizeBackend.pytorch,
                dtype=torch.float32,
                intermediate_dtype=torch.float32,
            )
            captures[phase][name] = {
                "input": inputs.detach().cpu().clone(),
                "quantizer_input": normalized.detach().cpu().clone(),
                "decoded_activation": decoded.detach().cpu().clone(),
                "output": output.detach().cpu().clone(),
                "input_stride": list(inputs.stride()),
                "quantizer_input_stride": list(normalized.stride()),
                "row_scales": scales.detach().cpu().clone()
                if scales is not None
                else None,
            }

        runtime.observer = observe
    try:
        with use_forwards(original_forwards), torch.no_grad():
            observed_eager = float(loss_forward(batch).detach())
        phase = "compiled"
        with torch.no_grad():
            observed_compiled = float(loss_forward(batch).detach())
    finally:
        for runtime, observer in observers:
            runtime.observer = observer
    valid = (
        abs(observed_eager - eager_loss) <= 1e-6
        and abs(observed_compiled - compiled_loss) <= 1e-6
    )
    comparisons = []
    for name, _ in selected:
        eager, compiled = captures["eager"][name], captures["compiled"][name]
        comparison = {"module": name, "tensors": {}}
        for key in [
            "input",
            "quantizer_input",
            "decoded_activation",
            "output",
            "row_scales",
        ]:
            if eager[key] is not None:
                comparison["tensors"][key] = tensor_difference(
                    eager[key], compiled[key]
                )
        comparison["strides"] = {
            key: {"eager": eager[key], "compiled": compiled[key]}
            for key in ["input_stride", "quantizer_input_stride"]
        }
        comparisons.append(comparison)
    return {
        "instrumentation_preserves_losses": valid,
        "observed_eager_loss": observed_eager,
        "observed_compiled_loss": observed_compiled,
        "modules": comparisons,
    }
