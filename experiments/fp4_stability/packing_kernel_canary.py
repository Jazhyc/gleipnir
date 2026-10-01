"""Check exact packed activation parity and timing before model integration."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from dataclasses import replace
from pathlib import Path

import torch
from fouroversix import ModuleQuantizationConfig, fp4_matmul, quantize_to_fp4
from fouroversix.utils import MatmulBackend, QuantizeBackend

from gleipnir.fouroversix_training import FOUROVERSIX_VERSION
from gleipnir.fp4_quantization_kernels import quantize_activation_rows
from gleipnir.fp4_row_kernels import normalize_rows, rescale_rows


def measure(action, repeats: int = 30) -> dict[str, float]:
    """Time warmed calls with CUDA events and synchronized wall time."""
    for _ in range(5):
        action()
    torch.cuda.synchronize()
    start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    begin = time.perf_counter()
    start.record()
    for _ in range(repeats):
        action()
    stop.record()
    stop.synchronize()
    return {
        "cuda_ms": start.elapsed_time(stop) / repeats,
        "wall_ms": (time.perf_counter() - begin) * 1000 / repeats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--implementation", choices=["row", "tiled"], default="row")
    args = parser.parse_args()
    if importlib.metadata.version("fouroversix") != FOUROVERSIX_VERSION:
        raise RuntimeError("packing canary requires pinned FourOverSix 1.0.5")
    torch.manual_seed(0)
    config = ModuleQuantizationConfig(
        weight_scale_2d=True,
        scale_rule="mse",
        quantize_backend=QuantizeBackend.triton,
        matmul_backend=MatmulBackend.cutlass,
    )
    fixed_amax = torch.ones(1, device="cuda", dtype=torch.float32)
    activation_config = replace(
        config.get_activation_config(), kwargs={"x_amax": fixed_amax}
    )
    report = {
        "status": "running",
        "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "torch": torch.__version__,
        "triton": importlib.metadata.version("triton"),
        "fouroversix": FOUROVERSIX_VERSION,
        "implementation": args.implementation,
        "source_sha256": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in [
                "src/gleipnir/fp4_quantization_kernels.py",
                "src/gleipnir/fp4_row_kernels.py",
                "experiments/fp4_stability/packing_kernel_canary.py",
            ]
        },
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.output)

    save()
    try:
        for rows, columns in [
            (1, 256),
            (17, 320),
            (128, 2560),
            (513, 9216),
            (16384, 2560),
            (2048, 9216),
            (16384, 9216),
        ]:
            print(f"checking {(rows, columns)}", flush=True)
            values = torch.randn(rows, columns, device="cuda", dtype=torch.bfloat16)
            values[0] = 0
            if rows > 1:
                values[1] *= 1024
                values[-1, 0] = 65536
            normalized, expected_scales = normalize_rows(values)
            expected = quantize_to_fp4(normalized, activation_config)
            actual, scales = quantize_activation_rows(
                values, fixed_amax, implementation=args.implementation
            )
            checks = {
                "bitwise_row_scales": torch.equal(scales, expected_scales),
                "bitwise_fp4_values": torch.equal(actual.values, expected.values),
                "bitwise_fp8_scales": torch.equal(
                    actual.scale_factors.view(torch.uint8),
                    expected.scale_factors.view(torch.uint8),
                ),
            }
            case = {"shape": [rows, columns], **checks}
            if not all(checks.values()):
                for name, got, wanted in [
                    ("values", actual.values, expected.values),
                    (
                        "block_scales",
                        actual.scale_factors.view(torch.uint8),
                        expected.scale_factors.view(torch.uint8),
                    ),
                ]:
                    different = (got != wanted).flatten()
                    indices = torch.nonzero(different).flatten()[:16]
                    case[f"{name}_mismatch_count"] = int(different.sum())
                    case[f"{name}_mismatch_sample"] = {
                        "indices": indices.tolist(),
                        "actual": got.flatten()[indices].tolist(),
                        "expected": wanted.flatten()[indices].tolist(),
                    }
            report["cases"].append(case)
            save()
            if not all(checks.values()):
                raise ValueError(f"packed parity failed at {(rows, columns)}: {checks}")
            weight = torch.randn(512, columns, device="cuda", dtype=torch.bfloat16)
            packed_weight = quantize_to_fp4(weight, config.get_weight_config())
            expected_output = rescale_rows(
                fp4_matmul(expected, packed_weight, backend=MatmulBackend.cutlass),
                expected_scales,
            )
            actual_output = rescale_rows(
                fp4_matmul(actual, packed_weight, backend=MatmulBackend.cutlass), scales
            )
            case["bitwise_rescaled_cutlass"] = torch.equal(
                actual_output, expected_output
            )
            if not case["bitwise_rescaled_cutlass"]:
                raise ValueError(f"CUTLASS output parity failed at {(rows, columns)}")

            def reference(values=values):
                normalized_inputs, row_scales = normalize_rows(values)
                return quantize_to_fp4(normalized_inputs, activation_config), row_scales

            case["reference"] = measure(reference)
            case["fused"] = measure(
                lambda values=values: quantize_activation_rows(
                    values, fixed_amax, implementation=args.implementation
                )
            )
            case["cuda_speedup"] = (
                case["reference"]["cuda_ms"] / case["fused"]["cuda_ms"]
            )
            save()
            print(json.dumps(case), flush=True)
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
