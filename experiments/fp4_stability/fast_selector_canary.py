"""Measure FP16 selector disagreements, error, determinism and native timing."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from dataclasses import replace
from pathlib import Path

import torch
from fouroversix import ModuleQuantizationConfig, fp4_matmul, quantize_to_fp4
from fouroversix.quantize import dequantize
from fouroversix.utils import MatmulBackend, QuantizeBackend

from experiments.fp4_stability.packing_kernel_canary import measure
from gleipnir.fouroversix_training import FOUROVERSIX_VERSION
from gleipnir.fp4_fast_selector import (
    FP16_SELECTOR_TIE_RELATIVE_BAND,
    quantize_normalized_fp16_selector,
)
from gleipnir.fp4_row_kernels import normalize_rows, rescale_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if importlib.metadata.version("fouroversix") != FOUROVERSIX_VERSION:
        raise RuntimeError("FP16 selector canary requires pinned FourOverSix 1.0.5")
    torch.manual_seed(0)
    config = ModuleQuantizationConfig(
        weight_scale_2d=True,
        scale_rule="mse",
        quantize_backend=QuantizeBackend.triton,
        matmul_backend=MatmulBackend.cutlass,
    )
    activation = replace(
        config.get_activation_config(),
        kwargs={"x_amax": torch.ones(1, device="cuda", dtype=torch.float32)},
    )
    report = {
        "status": "running",
        "gpu": torch.cuda.get_device_name(),
        "fouroversix": FOUROVERSIX_VERSION,
        "strict_tie_guard_relative_band": FP16_SELECTOR_TIE_RELATIVE_BAND,
        "selector": "fp16_candidate_product_fp32_scaled_target_and_mse",
        "upstream_arithmetic_reference": "NVIDIA/TransformerEngine PR3068",
        "upstream_merge_commit": "b972fa899eddf69fa7812736d24e479e23a83d3d",
        "bitwise_transformer_engine_parity_claimed": False,
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        for rows, width in [
            (1, 256),
            (17, 320),
            (128, 2560),
            (513, 9216),
            (16384, 2560),
            (2048, 9216),
            (16384, 9216),
        ]:
            inputs = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
            inputs[0] = 0
            if rows > 1:
                inputs[1] *= 1024
                inputs[-1, 0] = 65536
            normalized, scales = normalize_rows(inputs)
            strict = quantize_to_fp4(normalized, activation)
            fast = quantize_normalized_fp16_selector(normalized, activation)
            repeated = quantize_normalized_fp16_selector(normalized, activation)
            pm, pn = strict.padded_shape

            def logical_scales(packed, pm=pm, pn=pn, rows=rows, width=width):
                return (
                    packed.scale_factors.view(torch.uint8)
                    .reshape(pm // 128, pn // 64, 32, 4, 4)
                    .permute(0, 3, 2, 1, 4)
                    .reshape(pm, pn // 16)[:rows, : width // 16]
                )

            disagreements = int((logical_scales(strict) != logical_scales(fast)).sum())
            agreement = 1 - disagreements / (rows * (width // 16))

            def decode(packed):
                return dequantize(
                    packed,
                    backend=QuantizeBackend.pytorch,
                    dtype=torch.float32,
                    intermediate_dtype=torch.float32,
                )

            strict_decoded, fast_decoded = decode(strict), decode(fast)
            target = normalized.float()
            strict_sse = float((strict_decoded - target).square().sum())
            fast_sse = float((fast_decoded - target).square().sum())
            error_ratio = fast_sse / strict_sse if strict_sse else 1.0
            weight = torch.randn(512, width, device="cuda", dtype=torch.bfloat16)
            packed_weight = quantize_to_fp4(weight, config.get_weight_config())
            strict_output = rescale_rows(
                fp4_matmul(strict, packed_weight, backend=MatmulBackend.cutlass), scales
            )
            fast_output = rescale_rows(
                fp4_matmul(fast, packed_weight, backend=MatmulBackend.cutlass), scales
            )
            output_error = (
                float(
                    (fast_output.float() - strict_output.float()).norm()
                    / strict_output.float().norm()
                )
                if rows > 1
                else 0.0
            )
            deterministic = torch.equal(fast.values, repeated.values) and torch.equal(
                fast.scale_factors.view(torch.uint8),
                repeated.scale_factors.view(torch.uint8),
            )
            checks = {
                "choice_agreement_passed": agreement >= 0.999,
                "error_increase_passed": error_ratio <= 1.001,
                "output_relative_l2_passed": output_error <= 0.002,
                "deterministic_packed_operands": deterministic,
                "finite_output": bool(torch.isfinite(fast_output).all()),
            }
            case = {
                "shape": [rows, width],
                "scale_disagreement_count": disagreements,
                "scale_agreement_fraction": agreement,
                "strict_sse": strict_sse,
                "fast_sse": fast_sse,
                "sse_ratio": error_ratio,
                "output_relative_l2": output_error,
                **checks,
            }
            report["cases"].append(case)
            save()
            if not all(checks.values()):
                raise ValueError(f"FP16 selector gate failed: {case}")
            case["strict_packing_timing"] = measure(
                lambda values=normalized: quantize_to_fp4(values, activation)
            )
            case["fast_packing_timing"] = measure(
                lambda values=normalized: quantize_normalized_fp16_selector(
                    values, activation
                )
            )
            print(json.dumps(case), flush=True)
            save()
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
