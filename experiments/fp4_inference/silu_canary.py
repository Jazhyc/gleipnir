"""Validate fused FP8 activation packing on disjoint real MLP inputs."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

import torch
from vllm import _custom_ops as ops

from experiments.fp4_inference.kernel_canary import checked_load, relative_error
from experiments.local_inference.core import write_json
from experiments.local_inference.gemm_bench import telemetry
from gleipnir.qwen35_adapter_rebase import sha256_file
from gleipnir.silu_fp8 import silu_fp8


def stock_pack(gate_up: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Measure the stock two-kernel path including both output allocations."""
    activation = gate_up.new_empty((gate_up.shape[0], gate_up.shape[1] // 2))
    torch.ops._C.silu_and_mul(activation, gate_up)
    return ops.scaled_fp8_quant(activation, use_per_token_if_dynamic=True)


def complete(
    sample: torch.Tensor,
    weight_q: torch.Tensor,
    weight_s: torch.Tensor,
    pack: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    activations, scales = pack(sample)
    return ops.cutlass_scaled_mm(
        activations,
        weight_q,
        scale_a=scales,
        scale_b=weight_s,
        out_dtype=torch.bfloat16,
    )


def timing(
    call: Callable[[], torch.Tensor], heat: Callable[[], torch.Tensor], calls: int
) -> dict:
    for _ in range(5):
        call()
    until = time.monotonic() + 3
    while time.monotonic() < until:
        heat()
    torch.cuda.synchronize()
    before = telemetry()
    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    start.record()
    for _ in range(calls):
        call()
    end.record()
    end.synchronize()
    return {
        "milliseconds": start.elapsed_time(end) / calls,
        "telemetry_before": before,
        "telemetry_after": telemetry(),
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture", type=Path, default=Path("results/fp4_inference/capture")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calls", type=int, default=256)
    args = parser.parse_args()
    if args.calls < 1:
        parser.error("calls must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    manifest_path = args.capture / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["state"] != "complete":
        raise ValueError("Capture incomplete")
    weights = checked_load(args.capture / "weights.pt", manifest["weights_sha256"])
    checksums = {item["row"]: item["sha256"] for item in manifest["completed"]}
    rows = [
        checked_load(args.capture / f"row_{i}.pt", checksums[i])
        for i, row in enumerate(manifest["rows"])
        if row["split"] == "calibration"
    ]
    result = {
        "state": "running",
        "source_sha256": sha256_file(Path(__file__)),
        "kernel_sha256": sha256_file(Path("src/gleipnir/kernels/silu_fp8.py")),
        "capture_manifest_sha256": sha256_file(manifest_path),
        "calls_per_window": args.calls,
        "conditions": [],
        "limits": {"packing_relative_l2": 0.005, "native_relative_l2": 0.005},
        "note": "One window per path; online allocations/activation/packing/GEMM "
        "included; offline weight packing and gate/up projection excluded. "
        "No monitor score claim.",
    }
    write_json(args.output / "result.json", result)
    for layer in (0, 16, 31):
        x = torch.cat([r[f"{layer}_gate_up"] for r in rows]).cuda()
        gate_weight = weights[f"{layer}_gate_up"].cuda()
        down_weight = weights[f"{layer}_down"].cuda()
        gate_up = torch.nn.functional.linear(x, gate_weight).contiguous()
        weight_q, weight_s = ops.scaled_fp8_quant(
            down_weight, use_per_token_if_dynamic=True
        )
        weight_q = weight_q.t()
        for count in (128, 2048):
            sample = gate_up[:count].contiguous()
            q, s = silu_fp8(sample)
            old_q, old_s = stock_pack(sample)
            decoded = q.float() * s
            old_decoded = old_q.float() * old_s
            pack_error = relative_error(decoded, old_decoded)
            output = complete(sample, weight_q, weight_s, silu_fp8)
            reference = decoded @ (weight_q.float() * weight_s.reshape(1, -1))
            native_error = relative_error(output, reference)
            if not torch.isfinite(decoded).all() or not torch.isfinite(output).all():
                raise ValueError("Nonfinite fused packing or native GEMM")
            if max(pack_error, native_error) > 0.005:
                raise ValueError(
                    f"Fused FP8 validation failed: {pack_error}, {native_error}"
                )
            heat = partial(torch.nn.functional.linear, x, gate_weight)
            measurements = {}
            for name, pack in (("stock", stock_pack), ("fused", silu_fp8)):
                measurements[name] = timing(
                    partial(complete, sample, weight_q, weight_s, pack),
                    heat,
                    args.calls,
                )
            condition = {
                "layer": layer,
                "rows": count,
                "packing_relative_l2": pack_error,
                "native_relative_l2": native_error,
                "complete_speedup": measurements["stock"]["milliseconds"]
                / measurements["fused"]["milliseconds"],
                "timing": measurements,
            }
            result["conditions"].append(condition)
            write_json(args.output / "result.json", result)
            print(json.dumps(condition), flush=True)
    result["state"] = "complete"
    write_json(args.output / "result.json", result)


if __name__ == "__main__":
    main()
