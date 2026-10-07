"""Validate and time native FP4 paths on disjoint real layer-0 activations."""

from __future__ import annotations

import argparse
import json
import random
import time
from functools import partial
from pathlib import Path

import torch
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import swizzle_blockscale

from experiments.local_inference.core import write_json
from experiments.local_inference.gemm_bench import telemetry
from gleipnir.nvfp4_reference import decode_nvfp4
from gleipnir.qwen35_adapter_rebase import sha256_file
from gleipnir.vllm_nvfp4 import NVFP4_MAX, native_linear, pack_weight


def checked_load(path: Path, checksum: str) -> dict:
    if sha256_file(path) != checksum:
        raise ValueError(f"Changed capture artifact: {path}")
    return torch.load(path, map_location="cpu", weights_only=True)


def decode(
    packed: torch.Tensor, blocks: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    values = decode_nvfp4(
        packed.cpu().numpy(), blocks.float().cpu().numpy(), scale.item()
    )
    return torch.from_numpy(values).to(packed.device)


def relative_error(output: torch.Tensor, reference: torch.Tensor) -> float:
    return (
        (output.float() - reference).norm() / reference.norm().clamp_min(1e-12)
    ).item()


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture", type=Path, default=Path("results/fp4_inference/capture")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["cutlass", "b12x"], required=True)
    parser.add_argument("--packer", choices=["cuda", "triton"], default="cuda")
    parser.add_argument(
        "--scale-mode", choices=["dynamic", "power2"], default="dynamic"
    )
    parser.add_argument("--calls", type=int, default=256)
    args = parser.parse_args()
    if args.calls < 1:
        parser.error("calls must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest_path = args.capture / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["state"] != "complete":
        raise ValueError("Capture incomplete")
    weights = checked_load(args.capture / "weights.pt", manifest["weights_sha256"])
    checksums = {item["row"]: item["sha256"] for item in manifest["completed"]}
    selected = [
        i for i, row in enumerate(manifest["rows"]) if row["split"] == "calibration"
    ]
    rows = [checked_load(args.capture / f"row_{i}.pt", checksums[i]) for i in selected]
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    result = {
        "state": "running",
        "backend": args.backend,
        "packer": args.packer,
        "activation_scale_mode": args.scale_mode,
        "capture_manifest_sha256": sha256_file(manifest_path),
        "source_sha256": sha256_file(Path(__file__)),
        "native_method_sha256": sha256_file(Path("src/gleipnir/serving/vllm/nvfp4.py")),
        "packer_source_sha256": sha256_file(
            Path("src/gleipnir/kernels/fp4/nvfp4_pack.py")
        ),
        "calls": args.calls,
        "conditions": [],
        "note": "One 256-call window; activation range/packing included in FP4; "
        "offline weight preparation excluded. Real calibration activations; "
        "no judge data.",
    }
    write_json(args.output / "result.json", result)
    for key in ("0_gate_up", "0_down"):
        x = torch.cat([row[key] for row in rows]).cuda().contiguous()
        weight = weights[key].cuda().contiguous()
        packed, blocks, global_scale = pack_weight(weight, args.packer)
        swizzled = swizzle_blockscale(blocks)
        xs = x.abs().amax().float().clamp_min(1e-12).reshape(1) / NVFP4_MAX
        if args.scale_mode == "power2":
            xs = torch.exp2(torch.ceil(torch.log2(xs)))
        if args.packer == "cuda":
            xpacked, xblocks = ops.scaled_fp4_quant(x, xs.reciprocal(), False)
        else:
            from gleipnir.nvfp4_pack import pack_nvfp4

            xpacked, xblocks = pack_nvfp4(x, xs.reciprocal())
        quantized_reference = (
            decode(xpacked, xblocks, xs) @ decode(packed, blocks, global_scale).t()
        )
        original_reference = x.float() @ weight.float().t()
        output = native_linear(
            x,
            packed,
            swizzled,
            global_scale,
            args.backend,
            args.packer,
            args.scale_mode,
        )
        if not torch.isfinite(output).all().item():
            raise ValueError("Nonfinite native output")
        implementation_error = relative_error(output, quantized_reference)
        if implementation_error > 0.01:
            raise ValueError(
                f"Native NVFP4 implementation mismatch: {implementation_error}"
            )
        shape = {
            "projection": key,
            "shape_mnk": [x.shape[0], weight.shape[0], x.shape[1]],
            "quantized_reference_rel_l2": implementation_error,
            "bf16_reference_rel_l2": relative_error(output, original_reference),
            "offline_weight_bytes": packed.numel() + blocks.numel() + 4,
            "timings": {},
        }
        operations = {
            "bf16": partial(torch.nn.functional.linear, x, weight),
            "nvfp4": partial(
                native_linear,
                x,
                packed,
                swizzled,
                global_scale,
                args.backend,
                args.packer,
                args.scale_mode,
            ),
        }
        order = list(operations)
        random.Random(20260929).shuffle(order)
        for name in order:
            function = operations[name]
            heat = time.perf_counter()
            while time.perf_counter() - heat < 3:
                torch.nn.functional.linear(x, weight)
            for _ in range(5):
                function()
            torch.cuda.synchronize()
            before = telemetry()
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            wall = time.perf_counter()
            start.record()
            for _ in range(args.calls):
                function()
            end.record()
            end.synchronize()
            shape["timings"][name] = {
                "cuda_ms_per_call": start.elapsed_time(end) / args.calls,
                "wall_ms_per_call": (time.perf_counter() - wall) * 1000 / args.calls,
                "telemetry_before": before,
                "telemetry_after": telemetry(),
            }
        shape["speedup"] = (
            shape["timings"]["bf16"]["cuda_ms_per_call"]
            / shape["timings"]["nvfp4"]["cuda_ms_per_call"]
        )
        result["conditions"].append(shape)
        write_json(args.output / "result.json", result)
        print(json.dumps(shape), flush=True)
    result["state"] = "complete"
    write_json(args.output / "result.json", result)


if __name__ == "__main__":
    main()
