"""Check native outputs and FP32 LoRA gradients after reference-weight offload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gleipnir.fouroversix_training import FrozenFourOverSixLinear, native_runtime
from gleipnir.fp4_memory import offload_reference_weights


class AdapterLinear(torch.nn.Module):
    def __init__(self, *, activation_selector: str, compiler_visible: bool):
        super().__init__()
        original = torch.nn.Linear(
            2560, 512, device="cuda", bias=False, dtype=torch.bfloat16
        )
        original.requires_grad_(False)
        layer_type = FrozenFourOverSixLinear
        if compiler_visible:
            from gleipnir.fp4_compiler_ops import CompilerVisibleFourOverSixLinear

            layer_type = CompilerVisibleFourOverSixLinear
        self.base_layer = layer_type(
            original,
            native_runtime(
                original.weight,
                backward_mode="dequantized_bf16",
                row_scaled_activations=True,
                fused_row_scaling=True,
                activation_selector=activation_selector,
            ),
        )
        self.a = torch.nn.Parameter(torch.randn(128, 2560, device="cuda") * 0.01)
        self.b = torch.nn.Parameter(torch.randn(512, 128, device="cuda") * 0.01)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        adapter = torch.nn.functional.linear(
            torch.nn.functional.linear(inputs, self.a.to(torch.bfloat16)),
            self.b.to(torch.bfloat16),
        )
        return self.base_layer(inputs) + adapter * 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--activation-selector", choices=["strict", "fp16"], default="strict"
    )
    parser.add_argument("--compiler-visible-native", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(0)
    report = {"status": "running", "gpu": torch.cuda.get_device_name()}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        model = AdapterLinear(
            activation_selector=args.activation_selector,
            compiler_visible=args.compiler_visible_native,
        )
        values = torch.randn(513, 2560, device="cuda", dtype=torch.bfloat16)
        gradient = torch.randn(513, 512, device="cuda", dtype=torch.bfloat16)
        before_input = values.clone().requires_grad_()
        before = model(before_input)
        before.backward(gradient)
        before_grads = [
            before_input.grad.clone(),
            model.a.grad.clone(),
            model.b.grad.clone(),
        ]
        reference = torch.nn.functional.linear(values, model.base_layer.weight)
        runtime = model.base_layer.runtime
        packed_ptr = runtime.weight.values.data_ptr()
        decoded_ptr = runtime.dequantized_weight.data_ptr()
        report["offload"] = offload_reference_weights(model)
        model.zero_grad(set_to_none=True)
        after_input = values.clone().requires_grad_()
        after = model(after_input)
        after.backward(gradient)
        after_grads = [after_input.grad, model.a.grad, model.b.grad]
        relocated_reference = torch.nn.functional.linear(
            values, model.base_layer.weight.to(values.device)
        )
        checks = {
            "exact_native_outputs": torch.equal(before, after),
            "exact_input_and_adapter_gradients": all(
                torch.equal(a, b)
                for a, b in zip(before_grads, after_grads, strict=True)
            ),
            "exact_dense_reference_probe": torch.equal(reference, relocated_reference),
            "fp32_master_adapters_preserved": model.a.dtype
            == model.b.dtype
            == torch.float32,
            "reference_weight_on_cpu": model.base_layer.weight.device.type == "cpu",
            "packed_and_decoded_caches_unchanged": runtime.weight.values.data_ptr()
            == packed_ptr
            and runtime.dequantized_weight.data_ptr() == decoded_ptr,
            "positive_reference_bytes_released": report["offload"][
                "released_reference_weight_bytes"
            ]
            > 0,
            "finite_outputs_and_gradients": all(
                bool(torch.isfinite(value).all())
                for value in [before, after, *before_grads, *after_grads]
            ),
        }
        report["checks"] = checks
        if not all(checks.values()):
            raise ValueError(f"reference offload gate failed: {checks}")
        report["status"] = "passed"
        print(json.dumps(checks), flush=True)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
