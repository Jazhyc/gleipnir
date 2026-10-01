"""Execute native arbitrary-batch FP4 arithmetic and input-gradient canaries."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import torch

from gleipnir.fouroversix_training import (
    FrozenFourOverSixLinear,
    native_runtime,
    normalize_activation_rows,
)


def main() -> None:
    from fouroversix.quantize import dequantize, quantize_to_fp4
    from fouroversix.utils import QuantizeBackend, RoundStyle

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--backward-mode", choices=["fp4", "dequantized_bf16"], default="fp4"
    )
    parser.add_argument("--row-scaled-activations", action="store_true")
    parser.add_argument("--fused-row-scaling", action="store_true")
    parser.add_argument("--fused-activation-packing", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(0)
    layer = torch.nn.Linear(256, 512, bias=False, device="cuda", dtype=torch.bfloat16)
    layer.requires_grad_(False)
    runtime = native_runtime(
        layer.weight,
        backward_mode=args.backward_mode,
        row_scaled_activations=args.row_scaled_activations,
        fused_row_scaling=args.fused_row_scaling,
        fused_activation_packing=args.fused_activation_packing,
    )
    stochastic = runtime.gradient_config
    runtime.gradient_config = dataclasses.replace(
        stochastic, round_style=RoundStyle.nearest
    )
    native = FrozenFourOverSixLinear(layer, runtime)
    results = []
    for batch_size in [1, 2, 4, 8]:
        inputs = torch.randn(
            batch_size, 17, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        grad = torch.randn(batch_size, 17, 512, device="cuda", dtype=torch.bfloat16)
        output = native(inputs)
        output.backward(grad)
        quantizer_inputs = inputs.detach().reshape(-1, 256)
        scales = None
        if args.row_scaled_activations:
            quantizer_inputs, scales = normalize_activation_rows(quantizer_inputs)
        xq = quantize_to_fp4(quantizer_inputs, runtime.activation_config)
        gq = quantize_to_fp4(grad.reshape(-1, 512), runtime.gradient_config)

        def decoded(tensor):
            return dequantize(
                tensor,
                backend=QuantizeBackend.pytorch,
                dtype=torch.float32,
                intermediate_dtype=torch.float32,
            )

        expected = decoded(xq) @ decoded(runtime.weight).T
        if scales is not None:
            expected *= scales
        expected_grad = (
            grad.reshape(-1, 512).float() @ runtime.dequantized_weight.float()
            if runtime.dequantized_weight is not None
            else decoded(gq) @ decoded(runtime.transposed_weight).T
        )
        forward_error = float(
            (output.detach().float().reshape(-1, 512) - expected).norm()
            / expected.norm()
        )
        backward_error = float(
            (inputs.grad.float().reshape(-1, 256) - expected_grad).norm()
            / expected_grad.norm()
        )
        if max(forward_error, backward_error) > 0.02:
            raise ValueError(
                f"native arithmetic gate failed: {forward_error}, {backward_error}"
            )
        results.append(
            {
                "batch_size": batch_size,
                "forward_relative_l2": forward_error,
                "backward_relative_l2": backward_error,
            }
        )
    runtime.gradient_config = stochastic
    inputs = torch.randn(
        8, 19, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    native(inputs).float().square().mean().backward()
    if inputs.grad is None or not bool(torch.isfinite(inputs.grad).all()):
        raise ValueError("configured input-gradient path produced nonfinite values")
    torch.cuda.synchronize()
    import fouroversix
    import fouroversix._C

    row_independence = None
    if args.row_scaled_activations:
        values = torch.randn(17, 256, device="cuda", dtype=torch.bfloat16)
        alone = native(values)
        together = native(torch.cat([values, torch.full_like(values, 1024)]))[:17]
        row_independence = float(
            (alone.float() - together.float()).norm() / alone.float().norm()
        )
        if row_independence > 0.02:
            raise ValueError(
                f"per-token quantization depends on other rows: {row_independence}"
            )
    report = {
        "status": "passed",
        "cases": results,
        "stochastic_backward_finite": True if args.backward_mode == "fp4" else None,
        "configured_backward_finite": True,
        "native_forward_calls": runtime.forward_calls,
        "native_backward_calls": runtime.backward_calls
        if args.backward_mode == "fp4"
        else 0,
        "dequantized_bf16_backward_calls": runtime.backward_calls
        if args.backward_mode == "dequantized_bf16"
        else 0,
        "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "package_path": fouroversix.__file__,
        "extension_path": fouroversix._C.__file__,
        "quantize_backend": "triton",
        "matmul_backend": "cutlass",
        "backward_mode": args.backward_mode,
        "row_scaled_activations": args.row_scaled_activations,
        "fused_row_scaling": args.fused_row_scaling,
        "fused_activation_packing": args.fused_activation_packing,
        "row_independence_relative_l2": row_independence,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
