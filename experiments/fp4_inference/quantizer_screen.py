"""Select NVFP4 clipping on disjoint calibration data, then audit held-out rows."""

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
from experiments.local_inference.core import write_json
from gleipnir.nvfp4_pack import pack_nvfp4
from gleipnir.nvfp4_reference import E2M1_VALUES, encode_e2m1
from gleipnir.qwen35_adapter_rebase import sha256_file
from gleipnir.vllm_nvfp4 import NVFP4_MAX

CLIPS = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75)


def unpack(
    packed: torch.Tensor, blocks: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Decode linear scales on GPU using the independent published code table."""
    lut = torch.tensor(E2M1_VALUES, device=packed.device)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(1).long()
    return lut[codes] * blocks.float().repeat_interleave(16, dim=1) * scale


def scaled_pack(
    x: torch.Tensor, clip: float, mode: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scale = x.abs().amax().float().clamp_min(1e-12).reshape(1) / NVFP4_MAX
    if mode == "power2":
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    packed, blocks = pack_nvfp4(x, scale.reciprocal(), clip=clip)
    return packed, blocks, scale


def validate_packer() -> dict:
    """Exhaustively check E2M1 ties, signs, padding and native GEMM arithmetic."""
    pattern = np.array(
        [
            6,
            0.25,
            0.75,
            1.25,
            1.75,
            2.5,
            3.5,
            5,
            -6,
            -0.25,
            -0.75,
            -1.25,
            -1.75,
            -2.5,
            -3.5,
            -0.0,
        ],
        dtype=np.float32,
    )
    source = np.tile(pattern, (5, 4))
    x = torch.from_numpy(source).cuda().bfloat16()
    scale = torch.ones(1, device="cuda")
    packed, blocks = pack_nvfp4(x, scale, clip=1.0)
    codes = encode_e2m1(source)
    expected = codes[:, 0::2] | (codes[:, 1::2] << 4)
    np.testing.assert_array_equal(packed.cpu().numpy(), expected)
    np.testing.assert_array_equal(blocks.float().cpu().numpy(), np.ones((5, 4)))
    _, swizzled = pack_nvfp4(x, scale, swizzled=True)
    assert torch.equal(
        swizzled.view(torch.uint8), swizzle_blockscale(blocks).view(torch.uint8)
    )
    return {"tie_sign_and_swizzle_check": "passed", "rows": 5, "width": 64}


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
        "capture_sha256": sha256_file(manifest_path),
        "source_sha256": sha256_file(Path(__file__)),
        "packer_sha256": sha256_file(Path("src/gleipnir/nvfp4_pack.py")),
        "validation": validate_packer(),
        "clips": list(CLIPS),
        "selection_rule": "lowest mean calibration relative L2 across six "
        "projections; held-out rows never select",
        "projections": {},
    }
    write_json(args.output / "result.json", record)
    started = time.perf_counter()
    for key, cpu_weight in weights.items():
        weight = cpu_weight.cuda().contiguous()
        entry = {"shape_nk": list(weight.shape), "splits": {}}
        stock_scale = weight.abs().amax().float().reshape(1) / NVFP4_MAX
        stock_packed, stock_blocks = ops.scaled_fp4_quant(
            weight, stock_scale.reciprocal(), False
        )
        custom_packed, custom_blocks = pack_nvfp4(weight, stock_scale.reciprocal())
        entry["unclipped_vs_stock_weight_rel_l2"] = relative_error(
            unpack(custom_packed, custom_blocks, stock_scale),
            unpack(stock_packed, stock_blocks, stock_scale),
        )
        if entry["unclipped_vs_stock_weight_rel_l2"] > 0.01:
            raise ValueError(f"Unclipped packer disagrees with stock packing: {key}")
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
            reference = x.float() @ weight.float().t()
            scores = []
            for mode in ("dynamic", "power2"):
                for wc in CLIPS:
                    wp, wb, ws = scaled_pack(weight, wc, mode)
                    wq = unpack(wp, wb, ws)
                    for ac in CLIPS:
                        xp, xb, xs = scaled_pack(x, ac, mode)
                        xq = unpack(xp, xb, xs)
                        native = ops.cutlass_scaled_fp4_mm(
                            xp,
                            wp,
                            swizzle_blockscale(xb),
                            swizzle_blockscale(wb),
                            xs * ws,
                            x.dtype,
                        )
                        quantized_reference = xq @ wq.t()
                        implementation = relative_error(native, quantized_reference)
                        if (
                            not torch.isfinite(native).all().item()
                            or implementation > 0.01
                        ):
                            raise ValueError(
                                f"Native arithmetic failure {key}: {implementation}"
                            )
                        scores.append(
                            {
                                "scale_mode": mode,
                                "weight_clip": wc,
                                "activation_clip": ac,
                                "relative_l2": relative_error(native, reference),
                                "implementation_relative_l2": implementation,
                                "activation_amax": x.abs().amax().item(),
                            }
                        )
            entry["splits"][split] = scores
        record["projections"][key] = entry
        write_json(args.output / "result.json", record)
        print(
            json.dumps({"projection": key, "seconds": time.perf_counter() - started}),
            flush=True,
        )
        del weight, reference, native, quantized_reference, xq, wq
    aggregate = {}
    for split in ("calibration", "heldout"):
        grouped = {}
        for entry in record["projections"].values():
            for score in entry["splits"][split]:
                identity = (
                    score["scale_mode"],
                    score["weight_clip"],
                    score["activation_clip"],
                )
                grouped.setdefault(identity, []).append(score["relative_l2"])
        aggregate[split] = [
            {
                "scale_mode": k[0],
                "weight_clip": k[1],
                "activation_clip": k[2],
                "mean_relative_l2": sum(v) / len(v),
            }
            for k, v in grouped.items()
        ]
    selected = min(aggregate["calibration"], key=lambda x: x["mean_relative_l2"])
    heldout = next(
        s
        for s in aggregate["heldout"]
        if all(
            s[k] == selected[k]
            for k in ("scale_mode", "weight_clip", "activation_clip")
        )
    )
    record.update(
        state="complete",
        aggregate=aggregate,
        selected=selected,
        selected_heldout=heldout,
        seconds=time.perf_counter() - started,
    )
    write_json(args.output / "result.json", record)
    print(json.dumps({"selected": selected, "heldout": heldout}), flush=True)


if __name__ == "__main__":
    main()
