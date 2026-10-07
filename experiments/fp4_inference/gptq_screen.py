"""Validate Hessian feedback independently and screen six disjoint projections."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import swizzle_blockscale

from experiments.fp4_inference.kernel_canary import checked_load, relative_error
from experiments.fp4_inference.quantizer_screen import scaled_pack, unpack
from experiments.local_inference.core import write_json
from gleipnir.nvfp4_gptq import hessian_factor, quantize_gptq
from gleipnir.nvfp4_reference import E2M1_VALUES, encode_e2m1
from gleipnir.qwen35_adapter_rebase import sha256_file
from gleipnir.vllm_nvfp4 import pack_weight


def cpu_feedback(
    weight: np.ndarray, upper: np.ndarray, global_scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """Small independent fixed-order NumPy reference; no GPU feedback code reused."""
    weight = weight.copy()
    rows, width = weight.shape
    codes = np.empty_like(weight, dtype=np.uint8)
    scales = np.empty((rows, width // 16), dtype=np.float32)
    for column in range(width):
        if column % 16 == 0:
            group = np.abs(weight[:, column : column + 16]).max(axis=1)
            scales[:, column // 16] = (
                torch.from_numpy(np.minimum(group / (6 * global_scale), 448))
                .to(torch.float8_e4m3fn)
                .float()
                .numpy()
            )
        sf = scales[:, column // 16] * np.float32(global_scale)
        current = weight[:, column].copy()
        code = encode_e2m1(current / np.where(sf > 0, sf, 1))
        quantized = E2M1_VALUES[code] * sf
        codes[:, column] = code
        error = (current - quantized) / upper[column, column]
        weight[:, column:] -= error[:, None] * upper[column, column:]
    return codes[:, 0::2] | (codes[:, 1::2] << 4), scales


def validate_feedback() -> dict:
    generator = torch.Generator(device="cuda").manual_seed(20260929)
    weight = torch.randn((32, 64), device="cuda", generator=generator).bfloat16()
    x = torch.randn((256, 64), device="cuda", generator=generator).bfloat16()
    factor, _ = hessian_factor(x)
    packed, scales, global_scale, _ = quantize_gptq(weight, x, block=64)
    expected_packed, expected_scales = cpu_feedback(
        weight.float().cpu().numpy(), factor.cpu().numpy(), global_scale.item()
    )
    np.testing.assert_array_equal(packed.cpu().numpy(), expected_packed)
    np.testing.assert_array_equal(scales.float().cpu().numpy(), expected_scales)
    return {"numpy_feedback_codes_and_scales": "passed", "shape_nk": [32, 64]}


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture", type=Path, default=Path("results/fp4_inference/capture")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest_path = args.capture / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["state"] != "complete":
        raise ValueError("Incomplete capture")
    checksums = {c["row"]: c["sha256"] for c in manifest["completed"]}
    rows = [
        checked_load(args.capture / f"row_{i}.pt", checksums[i])
        for i in range(len(manifest["rows"]))
    ]
    weights = checked_load(args.capture / "weights.pt", manifest["weights_sha256"])
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    record = {
        "state": "running",
        "validation": validate_feedback(),
        "capture_manifest_sha256": sha256_file(manifest_path),
        "source_sha256": sha256_file(Path(__file__)),
        "feedback_source_sha256": sha256_file(
            Path("src/gleipnir/kernels/fp4/nvfp4_gptq.py")
        ),
        "protocol": {"damping": 0.01, "block_size": 128, "act_order": False},
        "projections": {},
    }
    write_json(args.output / "result.json", record)
    for key, value in weights.items():
        weight = value.cuda().contiguous()
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
        started = time.perf_counter()
        gp, gb, gs, metadata = quantize_gptq(weight, xcal)
        metadata["fit_seconds"] = time.perf_counter() - started
        sp, sb, ss = pack_weight(weight)
        torch.save(
            {"packed": gp.cpu(), "blocks": gb.cpu(), "global_scale": gs.cpu()},
            args.output / f"{key}.pt",
        )
        entry = {
            "metadata": metadata,
            "sha256": sha256_file(args.output / f"{key}.pt"),
            "splits": {},
        }
        for split in ("calibration", "heldout"):
            x = (
                torch.cat(
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
            xq = unpack(xp, xb, xs)
            reference = x.float() @ weight.float().t()
            scores = {}
            for name, p, b, s in [("stock", sp, sb, ss), ("gptq", gp, gb, gs)]:
                decoded = unpack(p, b, s)
                native = ops.cutlass_scaled_fp4_mm(
                    xp,
                    p,
                    swizzle_blockscale(xb),
                    swizzle_blockscale(b),
                    xs * s,
                    x.dtype,
                )
                implementation = relative_error(native, xq @ decoded.t())
                if implementation > 0.01 or not torch.isfinite(native).all().item():
                    raise ValueError("GPTQ native arithmetic failure")
                scores[name] = {
                    "relative_l2": relative_error(native, reference),
                    "implementation_relative_l2": implementation,
                    "weight_only_relative_l2": relative_error(
                        x.float() @ decoded.t(), reference
                    ),
                }
            entry["splits"][split] = scores
        record["projections"][key] = entry
        write_json(args.output / "result.json", record)
        print(json.dumps({"projection": key, **entry}), flush=True)
    summary = {}
    for split in ("calibration", "heldout"):
        summary[split] = {
            name: sum(
                e["splits"][split][name]["relative_l2"]
                for e in record["projections"].values()
            )
            / len(weights)
            for name in ("stock", "gptq")
        }
    eligible = (
        summary["calibration"]["gptq"] <= 0.9 * summary["calibration"]["stock"]
        and summary["heldout"]["gptq"] <= 1.02 * summary["heldout"]["stock"]
    )
    record.update(state="complete", summary=summary, expand_to_all_layers=eligible)
    write_json(args.output / "result.json", record)
    print(json.dumps({"summary": summary, "expand": eligible}), flush=True)


if __name__ == "__main__":
    main()
