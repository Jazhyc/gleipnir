"""Native timing of calibrated standalone recipes, including transforms."""

# All closures are consumed before advancing to the next recipe/projection.
# ruff: noqa: B023
import argparse
import ctypes
import json
import random
import time
from pathlib import Path

import torch
import triton
import triton.language as tl

from experiments.int4_calibration.screen import error, hadamard, quant, rotation
from experiments.local_inference.core import write_json
from experiments.local_inference.gemm_bench import telemetry
from experiments.local_inference.int4_bench import unpack


@triton.jit
def fast_rotation(X, S, Y, BLOCK: tl.constexpr):
    j = tl.arange(0, BLOCK)
    v = tl.load(X + tl.program_id(0) * BLOCK + j).to(tl.float32) * tl.load(S + j)
    for stage in tl.static_range(0, (BLOCK.bit_length() - 1)):
        other = tl.gather(v, j ^ (1 << stage), axis=0)
        v = tl.where((j & (1 << stage)) == 0, v + other, other - v)
    tl.store(Y + tl.program_id(0) * BLOCK + j, v * (BLOCK**-0.5))


@triton.jit
def pack(
    X,
    Q,
    S,
    R: tl.constexpr,
    K: tl.constexpr,
    G: tl.constexpr,
    CLIP: tl.constexpr,
    B: tl.constexpr,
):
    row, group = tl.program_id(0), tl.program_id(1)
    j = tl.arange(0, B)
    x = tl.load(X + row * K + group * G + j, j < G, 0).to(tl.float32)
    s = tl.maximum(tl.div_rn(tl.max(tl.abs(x), 0) * CLIP, 7.0), 1e-12)
    tl.store(S + group * R + row, s)
    p = tl.arange(0, B // 2)
    a = tl.load(X + row * K + group * G + 2 * p, 2 * p < G, 0).to(tl.float32)
    b = tl.load(X + row * K + group * G + 2 * p + 1, 2 * p + 1 < G, 0).to(tl.float32)
    a = tl.extra.cuda.libdevice.nearbyint(tl.div_rn(a, s))
    b = tl.extra.cuda.libdevice.nearbyint(tl.div_rn(b, s))
    a = tl.minimum(7.0, tl.maximum(-7.0, a)).to(tl.int32)
    b = tl.minimum(7.0, tl.maximum(-7.0, b)).to(tl.int32)
    tl.store(
        Q + (group * R + row) * (G // 2) + p, (a & 15) | ((b & 15) << 4), p < G // 2
    )


@triton.jit
def combine(
    C,
    XS,
    WS,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    GROUPS: tl.constexpr,
    B: tl.constexpr,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    total = tl.full((B,), 0, tl.float32)
    for g in range(GROUPS):
        c = tl.load(C + g * M * N + i, i < M * N, 0).to(tl.float32)
        xs = tl.load(XS + g * M + i // N, i < M * N, 0)
        ws = tl.load(WS + g * N + i % N, i < M * N, 0)
        total = total + c * xs * ws
    tl.store(Y + i, total, i < M * N)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--fast-rotation", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.capture / "manifest.json").read_text())
    screen = json.loads((args.screen / "result.json").read_text())
    weights = torch.load(args.capture / "weights.pt", weights_only=True)
    rows = [
        torch.load(args.capture / f"row_{i}.pt", weights_only=True)
        for i, r in enumerate(manifest["rows"])
        if r["split"] == "calibration"
    ]
    lib = ctypes.CDLL(str(args.library.resolve()))
    fn = lib.int4_gemm
    fn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    fn.restype = ctypes.c_int
    torch.backends.cuda.matmul.allow_tf32 = False
    result = {
        "calls_per_window": 256,
        "windows": 1,
        "rows": [],
        "note": "Real calibration activations; standalone native kernels. "
        "Group path materializes INT32 partials; deliberately unfused prototype. "
        "Online transforms use PyTorch FP32; weight transforms offline.",
    }
    if args.previous:
        result["rows"] = json.loads((args.previous / "result.json").read_text())["rows"]
        result["recovered_from"] = str(args.previous)
    write_json(args.output / "protocol.json", result)
    for key in ("0_gate_up", "0_down"):
        x = torch.cat([r[key] for r in rows]).cuda()
        w = weights[key].cuda()
        m, k = x.shape
        n = w.shape[0]
        ref = x.float() @ w.float().t()
        bf = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

        def baseline():
            torch.mm(x, w.t(), out=bf)

        selections = next(
            r["selected"] for r in screen["projections"] if r["projection"] == key
        )
        families = ["bf16", "naive", "clip", "smooth", "rotate", "group"]
        if args.fast_rotation:
            families = ["rotate"]
        random.Random(20260924).shuffle(families)
        until = time.perf_counter() + 3
        while time.perf_counter() < until:
            for _ in range(32):
                baseline()
            torch.cuda.synchronize()
        for family in families:
            if any(
                r["projection"] == key and r["recipe"]["family"] == family
                for r in result["rows"]
            ):
                continue
            recipe = selections[family]["recipe"]
            if args.fast_rotation:
                recipe = dict(recipe, implementation="triton_fht")
            if family == "bf16":
                op = baseline
                baseline()
                metrics = error(bf, ref)
            else:
                scale = None
                h = signs = None
                if family == "smooth":
                    alpha = recipe["alpha"]
                    scale = (
                        x.float().abs().amax(0).clamp_min(1e-5) ** alpha
                        / w.float().abs().amax(0).clamp_min(1e-5) ** (1 - alpha)
                    ).clamp(0.001, 1000)
                if family == "rotate":
                    h = hadamard(recipe["block"])
                    gen = torch.Generator().manual_seed(20260924)
                    signs = (
                        torch.randint(0, 2, (recipe["block"],), generator=gen) * 2 - 1
                    ).cuda()
                    rotated = torch.empty_like(x, dtype=torch.float32)

                def transform():
                    if family == "smooth":
                        return x.float() / scale
                    if family == "rotate":
                        if args.fast_rotation:
                            fast_rotation[(m * k // recipe["block"],)](
                                x, signs, rotated, recipe["block"]
                            )
                            return rotated
                        return rotation(x, h, signs)
                    return x

                ww = w.float()
                if family == "smooth":
                    ww = ww * scale
                if family == "rotate":
                    ww = rotation(ww, h, signs)
                    if args.fast_rotation:
                        assert (
                            error(transform(), rotation(x, h, signs))["relative_l2"]
                            < 1e-5
                        )
                g = recipe.get("group", k)
                groups = k // g
                xq = torch.empty((groups, m, g // 2), device="cuda", dtype=torch.uint8)
                wq = torch.empty((groups, n, g // 2), device="cuda", dtype=torch.uint8)
                xs = torch.empty((groups, m), device="cuda")
                ws = torch.empty((groups, n), device="cuda")
                c = torch.empty((groups, m, n), device="cuda", dtype=torch.int32)
                y = torch.empty_like(bf)
                pack[(n, groups)](
                    ww,
                    wq,
                    ws,
                    n,
                    k,
                    g,
                    recipe.get("wclip", 1),
                    triton.next_power_of_2(g),
                )

                def op():
                    xx = transform()
                    pack[(m, groups)](
                        xx,
                        xq,
                        xs,
                        m,
                        k,
                        g,
                        recipe.get("xclip", 1),
                        triton.next_power_of_2(g),
                    )
                    for gi in range(groups):
                        status = fn(
                            xq[gi].data_ptr(),
                            wq[gi].data_ptr(),
                            c[gi].data_ptr(),
                            m,
                            n,
                            g,
                            0,
                            torch.cuda.current_stream().cuda_stream,
                        )
                        if status:
                            raise RuntimeError(f"CUTLASS status {status}")
                    combine[(triton.cdiv(m * n, 1024),)](
                        c, xs, ws, y, m, n, groups, 1024, enable_fp_fusion=False
                    )

                op()
                # Validate actual packing with explicitly rounded division.
                for original, packed, scales, clip in (
                    (transform().float(), xq, xs, recipe.get("xclip", 1)),
                    (ww, wq, ws, recipe.get("wclip", 1)),
                ):
                    z = original.reshape(-1, groups, g).permute(1, 0, 2)
                    sref = (
                        ((z.abs().amax(-1) * clip).double() / 7)
                        .float()
                        .clamp_min(1e-12)
                    )
                    qcheck = (
                        (z.double() / sref.double()[..., None])
                        .float()
                        .round()
                        .clamp(-7, 7)
                        .int()
                    )
                    assert torch.equal(scales, sref), "scale mismatch"
                    for gi in range(groups):
                        assert torch.equal(unpack(packed[gi]), qcheck[gi]), (
                            "pack mismatch"
                        )
                # Every partial integer GEMM must match, as must output scaling.
                scaled_reference = torch.zeros_like(y, dtype=torch.float32)
                for gi in range(groups):
                    qref = unpack(xq[gi]).float() @ unpack(wq[gi]).float().t()
                    assert torch.equal(c[gi].float(), qref), "integer mismatch"
                    scaled_reference += qref * xs[gi, :, None] * ws[gi, None, :]
                assert torch.equal(y, scaled_reference.bfloat16()), (
                    "output scaling mismatch"
                )
                numerical = (
                    quant(
                        transform().float(),
                        recipe.get("xclip", 1),
                        recipe.get("group", 0),
                    )
                    @ quant(ww, recipe.get("wclip", 1), recipe.get("group", 0)).t()
                )
                agreement = error(y, numerical)
                # Fake quantization uses PyTorch reciprocal scaling. Near rounding
                # ties it need not match the native div.rn recipe bit-for-bit.
                # Keep that discrepancy separate from exact native arithmetic checks.
                metrics = error(y, ref)
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
                "projection": key,
                "recipe": recipe,
                "gpu_ms": start.elapsed_time(end) / 256,
                "wall_ms": (time.perf_counter() - wall) * 1000 / 256,
                "error": metrics,
                "native_validation": "packing_integer_and_rescale_exact"
                if family != "bf16"
                else None,
                "simulation_difference": agreement if family != "bf16" else None,
                "before": before,
                "after": telemetry(),
            }
            result["rows"].append(row)
            write_json(args.output / "result.json", result)
            print(json.dumps(row), flush=True)
            if family != "bf16":
                del c, xq, wq, xs, ws, y, ww, qref, numerical
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
