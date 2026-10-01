"""Require exact native Inductor forward/dX behind opaque projection operators."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import torch

from gleipnir.fouroversix_training import FrozenFourOverSixLinear, native_runtime
from gleipnir.fp4_compiler_ops import CompilerVisibleFourOverSixLinear


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--activation-selector", choices=["strict", "fp16"], default="strict"
    )
    args = parser.parse_args()
    torch.manual_seed(0)
    report = {"status": "running", "gpu": torch.cuda.get_device_name(), "cases": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        for width in [256, 2560, 9216]:
            original = torch.nn.Linear(
                width, 512, bias=False, device="cuda", dtype=torch.bfloat16
            )
            original.requires_grad_(False)
            runtime = native_runtime(
                original.weight,
                backward_mode="dequantized_bf16",
                row_scaled_activations=True,
                fused_row_scaling=True,
                activation_selector=args.activation_selector,
            )
            reference = FrozenFourOverSixLinear(original, runtime)
            visible = CompilerVisibleFourOverSixLinear(
                original, dataclasses.replace(runtime)
            )
            compiled = torch.compile(
                visible, backend="inductor", fullgraph=True, dynamic=True
            )
            for batch_size in [1, 2, 4, 8]:
                values = torch.randn(
                    batch_size, 17, width, device="cuda", dtype=torch.bfloat16
                )
                expected_inputs = values.clone().requires_grad_()
                actual_inputs = values.clone().requires_grad_()
                expected, actual = reference(expected_inputs), compiled(actual_inputs)
                gradient = torch.randn_like(actual)
                expected.backward(gradient)
                actual.backward(gradient)
                checks = {
                    "exact_outputs": torch.equal(actual, expected),
                    "exact_input_gradients": torch.equal(
                        actual_inputs.grad, expected_inputs.grad
                    ),
                    "finite_gradients": bool(torch.isfinite(actual_inputs.grad).all()),
                }
                report["cases"].append(
                    {"width": width, "batch_size": batch_size, **checks}
                )
                save()
                if not all(checks.values()):
                    raise ValueError(f"compiler-visible native gate failed: {checks}")
                print(f"passed width={width} batch={batch_size}", flush=True)
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
