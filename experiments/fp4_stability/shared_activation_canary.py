"""Require exact native gate/up sharing across fresh and checkpointed calls."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint

from gleipnir.fouroversix_training import (
    FrozenFourOverSixLinear,
    PairedActivationCache,
    native_runtime,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(0)
    report = {"status": "running", "gpu": torch.cuda.get_device_name(), "cases": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        for rows, width in [(17, 256), (513, 2560), (2048, 9216)]:
            originals = [
                torch.nn.Linear(
                    width, 512, bias=False, device="cuda", dtype=torch.bfloat16
                )
                for _ in range(2)
            ]
            for original in originals:
                original.requires_grad_(False)
            runtimes = [
                native_runtime(
                    original.weight,
                    backward_mode="dequantized_bf16",
                    row_scaled_activations=True,
                    fused_row_scaling=True,
                )
                for original in originals
            ]
            shared = [dataclasses.replace(runtime) for runtime in runtimes]
            cache = PairedActivationCache()
            for runtime, role in zip(shared, ["gate", "up"], strict=True):
                runtime.activation_cache = cache
                runtime.activation_cache_role = role
            reference_layers = [
                FrozenFourOverSixLinear(o, r)
                for o, r in zip(originals, runtimes, strict=True)
            ]
            shared_layers = [
                FrozenFourOverSixLinear(o, r)
                for o, r in zip(originals, shared, strict=True)
            ]
            values = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
            values[0] = 0
            values[-1] *= 1024
            gradient = torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16)

            def action(layers, inputs):
                return torch.nn.functional.silu(layers[0](inputs)) * layers[1](inputs)

            for recompute in [False, True]:
                reference_input = values.clone().requires_grad_()
                shared_input = values.clone().requires_grad_()
                reference = action(reference_layers, reference_input)
                actual = (
                    checkpoint(
                        lambda x, layers=shared_layers: action(layers, x),
                        shared_input,
                        use_reentrant=False,
                    )
                    if recompute
                    else action(shared_layers, shared_input)
                )
                reference.backward(gradient)
                actual.backward(gradient)
                checks = {
                    "exact_outputs": torch.equal(reference, actual),
                    "exact_input_gradients": torch.equal(
                        reference_input.grad, shared_input.grad
                    ),
                    "finite_gradients": bool(torch.isfinite(shared_input.grad).all()),
                    "cache_consumed": cache.entry is None,
                    "one_pack_per_pair": shared[0].activation_pack_calls == cache.hits
                    and shared[1].activation_pack_calls == 0
                    and cache.misses == 0,
                }
                report["cases"].append(
                    {
                        "shape": [rows, width],
                        "checkpointed": recompute,
                        "cache_hits": cache.hits,
                        **checks,
                    }
                )
                save()
                if not all(checks.values()):
                    raise ValueError(f"shared native operand gate failed: {checks}")
                print(f"passed {(rows, width)} checkpointed={recompute}", flush=True)
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
