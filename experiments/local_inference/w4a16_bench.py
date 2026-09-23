"""One-window real-activation Marlin W4A16 versus CUTLASS FP8 screen."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import vllm
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    apply_gptq_marlin_linear,
    marlin_make_workspace_new,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    marlin_quantize,
)
from vllm.scalar_type import scalar_types

from experiments.local_inference.core import write_json
from experiments.local_inference.gemm_bench import telemetry
from gleipnir.qwen35_adapter_rebase import sha256_file


def checked_load(path: Path, expected: str) -> dict:
    if sha256_file(path) != expected:
        raise ValueError(f"Checksum mismatch: {path}")
    return torch.load(path, map_location="cpu", weights_only=True)


def relative_error(output: torch.Tensor, reference: torch.Tensor) -> float:
    return float((output.float() - reference).norm() / reference.norm())


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture", type=Path, default=Path("results/int4_calibration/capture")
    )
    parser.add_argument("--output", type=Path, required=True)
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
    hashes = {item["row"]: item["sha256"] for item in manifest["completed"]}
    selected = [
        i for i, row in enumerate(manifest["rows"]) if row["split"] == "calibration"
    ]
    rows = [checked_load(args.capture / f"row_{i}.pt", hashes[i]) for i in selected]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    result = {
        "state": "running",
        "torch": torch.__version__,
        "vllm": vllm.__version__,
        "device": torch.cuda.get_device_name(),
        "calls_per_window": args.calls,
        "windows_per_condition": 1,
        "capture_manifest_sha256": sha256_file(manifest_path),
        "script_sha256": sha256_file(Path(__file__)),
        "selected_rows": selected,
        "note": "Layer-0 isolated kernels; BF16 real captured activations. W4A16 uses "
        "round-to-nearest group quantization, not calibrated GPTQ/AWQ. FP8 includes "
        "dynamic per-token activation quantization and static per-channel weights. "
        "No whole-model quality or throughput claim. "
        "TF32 and reduced BF16 reduction off.",
        "shapes": [],
    }
    write_json(args.output / "result.json", result)
    for key in ("0_gate_up", "0_down"):
        x = torch.cat([row[key] for row in rows]).cuda().contiguous()
        w = weights[key].cuda().contiguous()
        n, k = w.shape
        if x.shape != (2048, k):
            raise ValueError(f"Unexpected activation shape: {x.shape}")
        reference = torch.nn.functional.linear(x.float(), w.float())
        w8, ws = ops.scaled_fp8_quant(w, use_per_token_if_dynamic=True)

        def fp8(x=x, w8=w8, ws=ws):
            x8, xs = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
            return ops.cutlass_scaled_mm(
                x8, w8.t(), xs, ws.t(), out_dtype=torch.bfloat16
            )

        candidates = {
            "bf16": lambda x=x, w=w: torch.nn.functional.linear(x, w),
            "fp8": fp8,
        }
        x8, xs = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
        quant_refs = {
            "bf16": reference,
            "fp8": (x8.float() * xs) @ (w8.float() * ws).t(),
        }
        storage = {
            "bf16": w.numel() * w.element_size(),
            "fp8": w8.numel() + ws.numel() * ws.element_size(),
        }
        for group in (128, 64):
            wr, packed, scales, gidx, sort, _ = marlin_quantize(
                w.t().contiguous(), scalar_types.uint4b8, group, act_order=False
            )
            workspace = marlin_make_workspace_new(x.device)
            zp = torch.empty(0, dtype=torch.int32, device=x.device)
            name = f"w4a16_g{group}"
            candidates[name] = (
                lambda p=packed, s=scales, g=gidx, idx=sort, wk=workspace, x=x, zp=zp, n=n, k=k: (  # noqa: E501
                    apply_gptq_marlin_linear(
                        x,
                        p,
                        s,
                        zp,
                        g,
                        idx,
                        wk,
                        scalar_types.uint4b8,
                        n,
                        k,
                        True,
                        input_dtype=torch.bfloat16,
                    )
                )
            )
            quant_refs[name] = x.float() @ wr.float()
            storage[name] = sum(t.numel() * t.element_size() for t in (packed, scales))
        shape = {"key": key, "mnk": [x.shape[0], n, k], "conditions": {}}
        result["shapes"].append(shape)
        for name, fn in candidates.items():
            y = fn()
            agreement = relative_error(y, quant_refs[name])
            if not bool(torch.isfinite(y).all()) or agreement > 0.005:
                raise ValueError(f"{name} quantized-math parity failed: {agreement}")
            shape["conditions"][name] = {
                "relative_l2_vs_quantized_fp32": agreement,
                "relative_l2_vs_original_fp32": relative_error(y, reference),
                "weight_and_scale_bytes": storage[name],
            }
        del quant_refs
        order = list(candidates)
        random.Random(20260924).shuffle(order)
        shape["order"] = order
        for name in order:
            fn = candidates[name]
            # Identical bounded preconditioning before each single measured window.
            until = time.perf_counter() + 3
            while time.perf_counter() < until:
                for _ in range(32):
                    candidates["bf16"]()
                torch.cuda.synchronize()
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            before = telemetry()
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            wall = time.perf_counter()
            start.record()
            for _ in range(args.calls):
                fn()
            end.record()
            end.synchronize()
            metrics = shape["conditions"][name]
            metrics.update(
                {
                    "cuda_ms": start.elapsed_time(end) / args.calls,
                    "wall_ms": (time.perf_counter() - wall) * 1000 / args.calls,
                    "telemetry_before": before,
                    "telemetry_after": telemetry(),
                }
            )
            write_json(args.output / "result.json", result)
            print(key, name, metrics, flush=True)
    result["state"] = "complete"
    write_json(args.output / "result.json", result)


if __name__ == "__main__":
    main()
