"""One-window BF16 MLP GEMM screen; no serving changes or dataset evaluation."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time
from pathlib import Path

import torch
import triton
import triton.language as tl
from safetensors import safe_open

from experiments.local_inference.core import ROOT, write_json


@triton.jit
def matmul_kernel(
    A,
    W,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for offset in range(tl.cdiv(K, BK)):
        kk = offset * BK + k
        a = tl.load(
            A + m[:, None] * K + kk[None, :],
            (m[:, None] < M) & (kk[None, :] < K),
            other=0,
        )
        w = tl.load(
            W + n[None, :] * K + kk[:, None],
            (n[None, :] < N) & (kk[:, None] < K),
            other=0,
        )
        acc = tl.dot(a, w, acc)
    tl.store(
        C + m[:, None] * N + n[None, :],
        acc.to(tl.bfloat16),
        (m[:, None] < M) & (n[None, :] < N),
    )


def telemetry() -> str:
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=temperature.gpu,clocks.current.sm,power.draw,"
            "clocks_throttle_reasons.sw_thermal_slowdown",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()


def weight(name: str) -> torch.Tensor:
    for shard in (ROOT / "merged_bf16").glob("*.safetensors"):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            if name in handle.keys():
                return handle.get_tensor(name)
    raise KeyError(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calls", type=int, default=256)
    parser.add_argument(
        "--projection", choices=("both", "gate_up", "down"), default="both"
    )
    parser.add_argument("--candidates", nargs="+")
    args = parser.parse_args()
    if args.calls < 1:
        parser.error("calls must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(20260923)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tiles = [
        (64, 64, 32, 4),
        (64, 64, 64, 4),
        (128, 64, 32, 4),
        (128, 64, 64, 4),
        (64, 128, 32, 4),
        (64, 128, 64, 4),
        (128, 128, 32, 8),
        (128, 128, 64, 8),
    ]
    specs = [
        ("torch_default", "default", True, None),
        ("torch_cublaslt", "cublaslt", True, None),
        ("torch_strict_cublas", "cublas", False, None),
        ("torch_strict_cublaslt", "cublaslt", False, None),
    ] + [
        (f"triton_{a}_{b}_{c}_w{d}", "default", False, (a, b, c, d))
        for a, b, c, d in tiles
    ]
    result = {
        "seed": 20260923,
        "calls_per_window": args.calls,
        "windows_per_candidate": 1,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "device": torch.cuda.get_device_name(),
        "specs": specs,
        "pid": os.getpid(),
        "selected_projection": args.projection,
        "selected_candidates": args.candidates,
        "shapes": [],
        "limits": {"relative_l2": 0.005, "max_abs_over_reference_rms": 0.05},
        "note": "Real layer-0 weights, synthetic normal BF16 activations. "
        "Triton uses FP32 accumulators. Default torch allows reduced-precision "
        "reductions; strict torch disables them. No judge-score parity claimed.",
    }
    write_json(args.output / "protocol.json", result)
    prefix = "model.language_model.layers.0.mlp."
    for projection in ("gate_up", "down"):
        if projection == "gate_up":
            w = torch.cat(
                [weight(prefix + p + ".weight") for p in ("gate_proj", "up_proj")]
            )
        else:
            w = weight(prefix + "down_proj.weight")
        w = w.to(device="cuda", dtype=torch.bfloat16).contiguous()
        n, k = w.shape
        m = 2048
        x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        if args.projection not in ("both", projection):
            continue
        torch.backends.cuda.preferred_blas_library("cublas")
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        reference = torch.nn.functional.linear(x.float(), w.float())
        reference_rms = reference.square().mean().sqrt()
        reference_norm = torch.linalg.vector_norm(reference)
        candidates = {}
        failures = []

        def configure(spec):
            torch.backends.cuda.preferred_blas_library(spec[1])
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = spec[2]

        def execute(spec, x=x, w=w, m=m, n=n, k=k):
            tile = spec[3]
            if tile is None:
                return torch.nn.functional.linear(x, w)
            bm, bn, bk, warps = tile
            out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
            matmul_kernel[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
                x, w, out, m, n, k, bm, bn, bk, num_warps=warps, num_stages=3
            )
            return out

        baseline = None
        for spec in specs:
            configure(spec)
            try:
                y = execute(spec)
                torch.cuda.synchronize()
                if baseline is None:
                    baseline = y.clone()
                error = y.float() - reference
                rel = (torch.linalg.vector_norm(error) / reference_norm).item()
                maximum = (error.abs().max() / reference_rms).item()
                metrics = {
                    "finite": bool(torch.isfinite(y).all()),
                    "relative_l2_vs_fp32": rel,
                    "max_abs_over_reference_rms": maximum,
                    "max_absolute_error_vs_fp32": error.abs().max().item(),
                    "relative_l2_vs_default": (
                        torch.linalg.vector_norm(y.float() - baseline.float())
                        / torch.linalg.vector_norm(baseline.float())
                    ).item(),
                    "passed": bool(torch.isfinite(y).all())
                    and rel <= 0.005
                    and maximum <= 0.05,
                }
                candidates[spec[0]] = metrics
            except (RuntimeError, triton.OutOfResources) as error:
                failures.append({"candidate": spec[0], "error": str(error)})
        write_json(args.output / f"{projection}_validation.json", candidates)
        print(projection, "validation", json.dumps(candidates), flush=True)
        if not candidates.get("torch_default", {}).get("finite", False):
            raise RuntimeError("Missing or nonfinite baseline; validation preserved")
        # Time the finite existing baseline even if the numerical gate rejects it;
        # its failure remains explicit. Never silently drop the timing denominator.
        # All candidates are compiled before measuring. Heat the GPU for 3 seconds.
        configure(specs[0])
        until = time.perf_counter() + 3
        while time.perf_counter() < until:
            for _ in range(32):
                execute(specs[0])
            torch.cuda.synchronize()
        order = list(specs)
        random.Random(20260923).shuffle(order)
        measurements = []
        for spec in order:
            if args.candidates and spec[0] not in args.candidates:
                continue
            if spec[0] not in candidates or (
                not candidates[spec[0]]["passed"] and spec[0] != "torch_default"
            ):
                continue
            configure(spec)
            for _ in range(5):
                execute(spec)
            torch.cuda.synchronize()
            before = telemetry()
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            wall = time.perf_counter()
            start.record()
            for _ in range(args.calls):
                execute(spec)
            end.record()
            end.synchronize()
            row = {
                "candidate": spec[0],
                "gpu_ms_per_call": start.elapsed_time(end) / args.calls,
                "wall_ms_per_call": (time.perf_counter() - wall) * 1000 / args.calls,
                "telemetry_before": before,
                "telemetry_after": telemetry(),
                **candidates[spec[0]],
            }
            measurements.append(row)
            print(projection, json.dumps(row), flush=True)
        base_ms = next(
            (
                r["gpu_ms_per_call"]
                for r in measurements
                if r["candidate"] == "torch_default"
            ),
            None,
        )
        for row in measurements:
            row["speedup_vs_default"] = (
                base_ms / row["gpu_ms_per_call"] if base_ms is not None else None
            )
        result["shapes"].append(
            {
                "projection": projection,
                "m": m,
                "n": n,
                "k": k,
                "measurements": measurements,
                "validation": candidates,
                "failures": failures,
            }
        )
        write_json(args.output / "result.json", result)
    print("GEMM screen complete", flush=True)


if __name__ == "__main__":
    main()
