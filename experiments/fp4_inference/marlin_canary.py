"""Screen native Marlin W4A16 FP4 at decode-like and prefill-like row counts."""

from __future__ import annotations

import argparse
import json
import random
import time
from functools import partial
from pathlib import Path

import torch
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
    is_fp4_marlin_supported,
    prepare_fp4_layer_for_marlin,
)

from experiments.fp4_inference.kernel_canary import checked_load, decode, relative_error
from experiments.local_inference.core import write_json
from experiments.local_inference.gemm_bench import telemetry
from gleipnir.qwen35_adapter_rebase import sha256_file
from gleipnir.vllm_nvfp4 import pack_weight


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if not is_fp4_marlin_supported():
        raise RuntimeError("Required Marlin FP4 kernel unavailable")
    root = Path("results/fp4_inference/capture")
    manifest = json.loads((root / "manifest.json").read_text())
    weights = checked_load(root / "weights.pt", manifest["weights_sha256"])
    checksums = {c["row"]: c["sha256"] for c in manifest["completed"]}
    rows = [
        checked_load(root / f"row_{i}.pt", checksums[i])
        for i, r in enumerate(manifest["rows"])
        if r["split"] == "calibration"
    ]
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    record = {
        "state": "running",
        "calls": 256,
        "source_sha256": sha256_file(Path(__file__)),
        "capture_sha256": sha256_file(root / "manifest.json"),
        "conditions": [],
        "note": "W4A16 with BF16 activations and native Marlin fused FP4 "
        "dequantization. Offline weight repacking excluded.",
    }
    write_json(args.output / "result.json", record)
    for key in ("0_gate_up", "0_down"):
        full_x = torch.cat([r[key] for r in rows]).cuda().contiguous()
        weight = weights[key].cuda().contiguous()
        packed, blocks, global_scale = pack_weight(weight)
        decoded = decode(packed, blocks, global_scale)
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(packed, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(blocks, requires_grad=False)
        layer.weight_global_scale = torch.nn.Parameter(
            global_scale, requires_grad=False
        )
        layer.output_size_per_partition, layer.input_size_per_partition = weight.shape
        layer.params_dtype = torch.bfloat16
        prepare_fp4_layer_for_marlin(layer)
        for count in (1, 128, 2048):
            x = full_x[:count]

            bf16 = partial(torch.nn.functional.linear, x, weight)
            marlin = partial(
                apply_fp4_marlin_linear,
                x,
                layer.weight,
                layer.weight_scale,
                layer.weight_global_scale,
                layer.workspace,
                weight.shape[0],
                weight.shape[1],
                use_fp32_reduce=True,
            )

            output = marlin()
            implementation = relative_error(output, x.float() @ decoded.t())
            if implementation > 0.005 or not torch.isfinite(output).all().item():
                raise ValueError(f"Marlin FP4 arithmetic failure: {implementation}")
            entry = {
                "projection": key,
                "shape_mnk": [count, weight.shape[0], weight.shape[1]],
                "implementation_relative_l2": implementation,
                "bf16_relative_l2": relative_error(
                    output, x.float() @ weight.float().t()
                ),
                "timings": {},
            }
            operations = {"bf16": bf16, "marlin_w4a16": marlin}
            order = list(operations)
            random.Random(20260929).shuffle(order)
            for name in order:
                heat = time.perf_counter()
                while time.perf_counter() - heat < 3:
                    torch.nn.functional.linear(full_x, weight)
                for _ in range(5):
                    operations[name]()
                torch.cuda.synchronize()
                before = telemetry()
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                wall = time.perf_counter()
                start.record()
                for _ in range(256):
                    operations[name]()
                end.record()
                end.synchronize()
                entry["timings"][name] = {
                    "cuda_ms": start.elapsed_time(end) / 256,
                    "wall_ms": (time.perf_counter() - wall) * 1000 / 256,
                    "telemetry_before": before,
                    "telemetry_after": telemetry(),
                }
            entry["speedup"] = (
                entry["timings"]["bf16"]["cuda_ms"]
                / entry["timings"]["marlin_w4a16"]["cuda_ms"]
            )
            record["conditions"].append(entry)
            write_json(args.output / "result.json", record)
            print(json.dumps(entry), flush=True)
    record["state"] = "complete"
    write_json(args.output / "result.json", record)


if __name__ == "__main__":
    main()
