"""Native W4A4 screen: real weights, synthetic activations, one timing window."""

# Shape-local closures are fully executed before advancing the shape loop.
# ruff: noqa: B023

import argparse
import ctypes
import hashlib
import json
import random
import time
from pathlib import Path

import torch
import triton
import triton.language as tl

from experiments.local_inference.core import write_json
from experiments.local_inference.gemm_bench import telemetry, weight


@triton.jit
def quantize(X, Q, S, K: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    j = tl.arange(0, B)
    x = tl.load(X + row * K + j, j < K, 0).to(tl.float32)
    scale = tl.div_rn(tl.maximum(tl.max(tl.abs(x), 0), 7e-12), 7.0)
    tl.store(S + row, scale)
    p = tl.arange(0, B // 2)
    a = tl.load(X + row * K + 2 * p, 2 * p < K, 0).to(tl.float32)
    b = tl.load(X + row * K + 2 * p + 1, 2 * p + 1 < K, 0).to(tl.float32)
    qa = tl.minimum(
        7.0, tl.maximum(-7.0, tl.extra.cuda.libdevice.nearbyint(tl.div_rn(a, scale)))
    ).to(tl.int32)
    qb = tl.minimum(
        7.0, tl.maximum(-7.0, tl.extra.cuda.libdevice.nearbyint(tl.div_rn(b, scale)))
    ).to(tl.int32)
    tl.store(Q + row * (K // 2) + p, (qa & 15) | ((qb & 15) << 4), p < K // 2)


@triton.jit
def rescale(C, XS, WS, Y, M: tl.constexpr, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    c = tl.load(C + i, i < M * N, 0).to(tl.float32)
    xs = tl.load(XS + i // N, i < M * N, 0)
    ws = tl.load(WS + i % N, i < M * N, 0)
    tl.store(Y + i, c * xs * ws, i < M * N)


def unpack(q: torch.Tensor) -> torch.Tensor:
    parts = torch.stack((q & 15, q >> 4), dim=-1).to(torch.int32)
    return torch.where(parts >= 8, parts - 16, parts).reshape(q.shape[0], -1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    lib = ctypes.CDLL(str(args.library.resolve()))
    fn = lib.int4_gemm
    fn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    fn.restype = ctypes.c_int
    torch.manual_seed(20260923)
    torch.backends.cuda.matmul.allow_tf32 = False
    result = {
        "seed": 20260923,
        "calls": 256,
        "windows": 1,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "device": torch.cuda.get_device_name(),
        "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
        "protocol": "Real layer-0 BF16 weights; synthetic normal BF16 activations. "
        "Symmetric [-7,7], per-row max-abs scales. Weights packed offline. "
        "Preallocated buffers for all paths; no CUDA graphs. One 256-call window "
        "per operation, randomized order, 3s BF16 warmup, 5 untimed warmup calls. "
        "Integer correctness exact; quantization errors descriptive, no quality pass.",
        "shapes": [],
    }
    write_json(args.output / "protocol.json", result)
    for projection in ("gate_up", "down"):
        prefix = "model.language_model.layers.0.mlp."
        w = (
            torch.cat(
                [weight(prefix + s + ".weight") for s in ("gate_proj", "up_proj")]
            )
            if projection == "gate_up"
            else weight(prefix + "down_proj.weight")
        )
        w = w.cuda().to(torch.bfloat16).contiguous()
        n, k = w.shape
        m = 2048
        x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        xq = torch.empty((m, k // 2), device="cuda", dtype=torch.uint8)
        wq = torch.empty((n, k // 2), device="cuda", dtype=torch.uint8)
        xs = torch.empty(m, device="cuda", dtype=torch.float32)
        ws = torch.empty(n, device="cuda", dtype=torch.float32)
        c = torch.empty((m, n), device="cuda", dtype=torch.int32)
        y = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        bf = torch.empty_like(y)
        quantize[(n,)](w, wq, ws, k, triton.next_power_of_2(k))

        def pack_x():
            quantize[(m,)](x, xq, xs, k, triton.next_power_of_2(k))

        def gemm(tile):
            status = fn(
                xq.data_ptr(),
                wq.data_ptr(),
                c.data_ptr(),
                m,
                n,
                k,
                tile,
                torch.cuda.current_stream().cuda_stream,
            )
            if status:
                raise RuntimeError(f"CUTLASS status {status}")

        def scale():
            rescale[(triton.cdiv(m * n, 1024),)](c, xs, ws, y, m, n, 1024)

        def full(tile):
            pack_x()
            gemm(tile)
            scale()

        def bf16():
            torch.mm(x, w.t(), out=bf)

        pack_x()
        # Validate packing against independently computed PyTorch rounding/scales.
        for original, packed, scales in ((x, xq, xs), (w, wq, ws)):
            # Float64 division then FP32 rounding avoids PyTorch scalar division's
            # reciprocal-multiply shortcut; mirrors explicitly rounded div.rn.
            ref_scale = (original.double().abs().amax(1).clamp_min(7e-12) / 7).float()
            assert torch.equal(scales, ref_scale), "scale mismatch"
            ref_q = (
                (original.double() / ref_scale.double()[:, None])
                .float()
                .round()
                .clamp(-7, 7)
                .int()
            )
            assert torch.equal(unpack(packed), ref_q), "packing mismatch"
        integer_ref = unpack(xq).float() @ unpack(wq).float().t()
        for tile in range(3):
            gemm(tile)
            assert torch.equal(c.float(), integer_ref), "integer GEMM mismatch"
        scale()
        ref_scaled = (integer_ref * xs[:, None] * ws[None, :]).to(torch.bfloat16)
        assert torch.equal(y, ref_scaled), "rescale mismatch"
        ref = x.float() @ w.float().t()
        bf16()
        errors = {}
        for name, output in (("int4", y), ("bf16", bf)):
            err = output.float() - ref
            assert torch.isfinite(output).all()
            errors[name] = {
                "relative_l2_vs_fp32": (err.norm() / ref.norm()).item(),
                "max_abs_over_reference_rms": (
                    err.abs().max() / ref.square().mean().sqrt()
                ).item(),
            }
        record = {
            "projection": projection,
            "m": m,
            "n": n,
            "k": k,
            "errors": errors,
            "integer_validation": "exact_all_three_tiles",
            "measurements": [],
        }
        result["shapes"].append(record)
        write_json(args.output / "result.json", result)
        print(projection, "validation", json.dumps(errors), flush=True)
        ops = {"bf16": bf16, "activation_quant_pack": pack_x, "output_rescale": scale}
        for tile in range(3):
            ops[f"int4_gemm_tile{tile}"] = lambda tile=tile: gemm(tile)
            ops[f"int4_full_tile{tile}"] = lambda tile=tile: full(tile)
        for op in ops.values():
            op()
        torch.cuda.synchronize()
        until = time.perf_counter() + 3
        while time.perf_counter() < until:
            for _ in range(32):
                bf16()
            torch.cuda.synchronize()
        order = list(ops)
        random.Random(20260923).shuffle(order)
        for name in order:
            op = ops[name]
            for _ in range(5):
                op()
            torch.cuda.synchronize()
            before = telemetry()
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            wall = time.perf_counter()
            start.record()
            for _ in range(256):
                op()
            end.record()
            end.synchronize()
            row = {
                "name": name,
                "gpu_ms": start.elapsed_time(end) / 256,
                "wall_ms": (time.perf_counter() - wall) * 1000 / 256,
                "before": before,
                "after": telemetry(),
            }
            record["measurements"].append(row)
            write_json(args.output / "result.json", result)
            print(projection, json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
