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
)


def main() -> None:
    from fouroversix.quantize import dequantize, quantize_to_fp4
    from fouroversix.utils import QuantizeBackend, RoundStyle

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(0)
    layer = torch.nn.Linear(256, 512, bias=False, device="cuda", dtype=torch.bfloat16)
    layer.requires_grad_(False)
    runtime = native_runtime(layer.weight)
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
        xq = quantize_to_fp4(
            inputs.detach().reshape(-1, 256), runtime.activation_config
        )
        gq = quantize_to_fp4(grad.reshape(-1, 512), runtime.gradient_config)

        def decoded(tensor):
            return dequantize(
                tensor,
                backend=QuantizeBackend.pytorch,
                dtype=torch.float32,
                intermediate_dtype=torch.float32,
            )

        expected = decoded(xq) @ decoded(runtime.weight).T
        expected_grad = decoded(gq) @ decoded(runtime.transposed_weight).T
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
        raise ValueError("stochastic native backward failed")
    torch.cuda.synchronize()
    import fouroversix
    import fouroversix._C

    report = {
        "status": "passed",
        "cases": results,
        "stochastic_backward_finite": True,
        "native_forward_calls": runtime.forward_calls,
        "native_backward_calls": runtime.backward_calls,
        "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "package_path": fouroversix.__file__,
        "extension_path": fouroversix._C.__file__,
        "quantize_backend": "triton",
        "matmul_backend": "cutlass",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
