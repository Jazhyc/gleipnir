"""Freeze all-MLP NVFP4 feedback artifacts after the predeclared numerical gate."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import swizzle_blockscale

from experiments.fp4_inference.gptq_screen import validate_feedback
from experiments.fp4_inference.kernel_canary import checked_load, relative_error
from experiments.fp4_inference.quantizer_screen import scaled_pack, unpack
from experiments.local_inference.core import write_json
from gleipnir.nvfp4_artifact import tensor_sha256
from gleipnir.nvfp4_gptq import quantize_gptq
from gleipnir.qwen35_adapter_rebase import sha256_file


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture", type=Path, default=Path("results/fp4_inference/capture_all")
    )
    parser.add_argument(
        "--screen", type=Path, default=Path("results/fp4_inference/gptq_screen")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    screen = json.loads((args.screen / "result.json").read_text())
    manifest_path = args.capture / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    prior_path = Path("results/fp4_inference/capture/manifest.json")
    prior = json.loads(prior_path.read_text())
    if screen["state"] != "complete" or not screen["expand_to_all_layers"]:
        raise ValueError("Predeclared GPTQ expansion gate did not pass")
    if sha256_file(prior_path) != screen["capture_manifest_sha256"]:
        raise ValueError("Changed screen calibration provenance")
    if manifest["state"] != "complete" or manifest["layers"] != list(range(32)):
        raise ValueError("All-layer capture missing or incomplete")
    for key in ("rows", "subset_sha256", "iteration32_sha256", "merge_manifest_sha256"):
        if manifest[key] != prior[key]:
            raise ValueError(f"Changed disjoint capture identity: {key}")
    weights = checked_load(args.capture / "weights.pt", manifest["weights_sha256"])
    prior_weights = checked_load(
        prior_path.parent / "weights.pt", prior["weights_sha256"]
    )
    for key, value in prior_weights.items():
        if not torch.equal(weights[key], value):
            raise ValueError(f"Changed original weights: {key}")
    del prior_weights
    checksums = {c["row"]: c["sha256"] for c in manifest["completed"]}
    rows = [
        checked_load(args.capture / f"row_{i}.pt", checksums[i])
        for i in range(len(manifest["rows"]))
    ]
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    record = {
        "state": "running",
        "validation": validate_feedback(),
        "capture_manifest_sha256": sha256_file(manifest_path),
        "screen_result_sha256": sha256_file(args.screen / "result.json"),
        "merge_manifest_sha256": manifest["merge_manifest_sha256"],
        "source_sha256": sha256_file(Path(__file__)),
        "feedback_source_sha256": sha256_file(Path("src/gleipnir/nvfp4_gptq.py")),
        "protocol": {"damping": 0.01, "block_size": 128, "act_order": False},
        "layers": {},
    }
    write_json(args.output / "manifest.json", record)
    for key, cpu_weight in weights.items():
        weight = cpu_weight.cuda().contiguous()
        xcal = (
            torch.cat(
                [
                    r[key]
                    for r, m in zip(rows, manifest["rows"], strict=True)
                    if m["split"] == "calibration"
                ]
            )
            .cuda()
            .contiguous()
        )
        start = time.perf_counter()
        packed, blocks, scale, metadata = quantize_gptq(weight, xcal)
        sample = unpack(packed[:16], blocks[:16], scale)
        reconstruction = relative_error(sample, weight[:16])
        if reconstruction > 0.25:
            raise ValueError(f"GPTQ source-weight reconstruction exceeds bound: {key}")
        splits = {}
        for split in ("calibration", "heldout"):
            x = (
                xcal
                if split == "calibration"
                else torch.cat(
                    [
                        r[key]
                        for r, m in zip(rows, manifest["rows"], strict=True)
                        if m["split"] == split
                    ]
                )
                .cuda()
                .contiguous()
            )
            xp, xb, xs = scaled_pack(x, 1.0, "dynamic")
            native = ops.cutlass_scaled_fp4_mm(
                xp,
                packed,
                swizzle_blockscale(xb),
                swizzle_blockscale(blocks),
                xs * scale,
                x.dtype,
            )
            implementation = relative_error(
                native, unpack(xp, xb, xs) @ unpack(packed, blocks, scale).t()
            )
            if implementation > 0.01 or not torch.isfinite(native).all().item():
                raise ValueError(f"GPTQ artifact native arithmetic failure: {key}")
            splits[split] = {
                "implementation_relative_l2": implementation,
                "bf16_output_relative_l2": relative_error(
                    native, x.float() @ weight.float().t()
                ),
            }
        path = args.output / f"{key}.pt"
        torch.save(
            {
                "packed": packed.cpu(),
                "blocks": blocks.cpu(),
                "global_scale": scale.cpu(),
            },
            path,
        )
        record["layers"][key] = {
            "file": path.name,
            "sha256": sha256_file(path),
            "shape_nk": list(weight.shape),
            "original_weight_sha256": tensor_sha256(cpu_weight),
            "metadata": metadata,
            "sample_weight_relative_l2": reconstruction,
            "splits": splits,
            "seconds": time.perf_counter() - start,
        }
        write_json(args.output / "manifest.json", record)
        print(
            json.dumps(
                {
                    "projection": key,
                    "completed": len(record["layers"]),
                    "seconds": record["layers"][key]["seconds"],
                }
            ),
            flush=True,
        )
    if len(record["layers"]) != 64:
        raise ValueError("Missing full-model MLP projection coverage")
    record["state"] = "complete"
    write_json(args.output / "manifest.json", record)


if __name__ == "__main__":
    main()
