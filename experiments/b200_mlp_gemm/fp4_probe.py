"""Native FP4 frozen-base forward/dgrad GEMMs, with packing included in timing."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import statistics
import time
import traceback
from collections.abc import Callable
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F

from gleipnir.bf16_lora import configure_bf16_reductions
from gleipnir.cudnn_fp4_gemm import (
    PACKING_KERNEL_METADATA,
    Nvfp4Gemm,
    decode_operand,
    pack_operand,
)


def relative_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30))


def measure(
    functions: dict[str, Callable], warmups: int = 6, repeats: int = 10
) -> dict:
    for _ in range(warmups):
        for fn in functions.values():
            fn()
    samples = {k: [] for k in functions}
    for repeat in range(repeats):
        names = list(functions)
        if repeat % 2:
            names.reverse()
        for name in names:
            torch.cuda.synchronize()
            start = time.perf_counter()
            functions[name]()
            torch.cuda.synchronize()
            samples[name].append((time.perf_counter() - start) * 1000)
    return {
        "samples_ms": samples,
        "mean_ms": {k: statistics.mean(v) for k, v in samples.items()},
    }


def graph_measure(
    x, w, b, op, *, row_scaled: bool = False, chunked_rows: bool = False
) -> dict:
    """Symmetric fixed-shape capture; include each caller-to-static input copy."""
    static_x = x.clone()

    def bf16():
        return F.linear(static_x, w)

    def fp4():
        return op(
            pack_operand(
                static_x,
                fused_amax=True,
                row_amax=row_scaled,
                chunked_rows=chunked_rows,
            ),
            b,
        )

    fns = {"bf16_graph": bf16, "fp4_graph_including_pack": fp4}
    graphs = {}
    outputs = {}
    warm_stream = torch.cuda.Stream()
    warm_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm_stream):
        for fn in fns.values():
            for _ in range(3):
                fn()
    torch.cuda.current_stream().wait_stream(warm_stream)
    torch.cuda.synchronize()
    for name, fn in fns.items():
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            outputs[name] = fn()
        graphs[name] = g

    def replay(g):
        static_x.copy_(x)
        g.replay()

    timings = measure({name: partial(replay, g) for name, g in graphs.items()})
    timings["copy_included"] = True
    timings["fp4_replay_relative_l2"] = relative_l2(
        outputs["fp4_graph_including_pack"], fp4()
    )
    # A changed input must recompute quantization; stale captured scales are unsafe.
    static_x.copy_(x * 0.5)
    graphs["fp4_graph_including_pack"].replay()
    torch.cuda.synchronize()
    timings["changed_input_relative_l2"] = relative_l2(
        outputs["fp4_graph_including_pack"], fp4()
    )
    if (
        timings["fp4_replay_relative_l2"] != 0
        or timings["changed_input_relative_l2"] != 0
    ):
        raise ValueError("FP4 CUDA graph replay changed dynamic quantization")
    means = timings["mean_ms"]
    timings["relative_improvement"] = (
        1 - means["fp4_graph_including_pack"] / means["bf16_graph"]
    )
    return timings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--optimized", action="store_true")
    parser.add_argument("--row-scaled", action="store_true")
    parser.add_argument("--chunked-rows", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = args.output / "probe.json"
    report = {
        "status": "starting",
        "scope": (
            "four isolated dense frozen-base GEMMs; "
            "excludes LoRA and activation derivatives"
        ),
        "synthetic_weights": True,
        "optimized": args.optimized,
        "row_scaled": args.row_scaled,
        "chunked_rows": args.chunked_rows,
        "full_training": False,
        "shapes": [],
        "quantization": {
            "format": "NVFP4",
            "weight_scaling": "16x16",
            "activation_scaling": "1x16",
            "global_scaling": "dynamic amax / (448*6)",
            "activation_amax_scope": "row" if args.row_scaled else "tensor",
            "rounding": "nearest even including dgrad",
            "stochastic_rounding": False,
            "decoded_oracle_l2_limit": 0.01,
        },
    }

    def save():
        receipt.write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        configure_bf16_reductions(allow_reduced_precision=False, allow_split_k=False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(41)
        report["runtime"] = {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "versions": {
                n: importlib.metadata.version(n)
                for n in ("nvidia-cudnn-frontend", "triton")
            },
        }
        h, i = 2560, 9216
        weights = {
            "gate_up": torch.randn(2 * i, h, dtype=torch.bfloat16, device="cuda")
            / math.sqrt(h),
            "down": torch.randn(h, i, dtype=torch.bfloat16, device="cuda")
            / math.sqrt(i),
        }
        packed = {}
        decoded = {}
        report["weights"] = {}
        for name, w in weights.items():
            torch.cuda.synchronize()
            start = time.perf_counter()
            packed[name] = pack_operand(w, weight=True)
            packed[name + "_dgrad"] = pack_operand(w.t().contiguous(), weight=True)
            torch.cuda.synchronize()
            pack_seconds = time.perf_counter() - start
            decoded[name] = decode_operand(packed[name])
            decoded[name + "_dgrad"] = decode_operand(packed[name + "_dgrad"])
            report["weights"][name] = {
                "pack_seconds_including_first_jit": pack_seconds,
                "decoded_transpose_relative_l2": relative_l2(
                    decoded[name + "_dgrad"], decoded[name].t()
                ),
                "quantization_relative_l2": relative_l2(decoded[name], w),
                "packed_pair_bytes": sum(
                    t.numel() * t.element_size()
                    for key in (name, name + "_dgrad")
                    for t in (
                        packed[key].codes,
                        packed[key].scales,
                        packed[key].inverse,
                    )
                ),
            }
            if report["weights"][name]["decoded_transpose_relative_l2"] != 0:
                raise ValueError("2D scaled forward/backward weights disagree")
            save()
        for m in (193, 4096, 16384):
            for name, original in weights.items():
                for backward in (False, True):
                    key = name + ("_dgrad" if backward else "")
                    w = original.t() if backward else original
                    n, k = w.shape
                    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
                    if backward:
                        x *= 0.001
                    a = pack_operand(
                        x,
                        fused_amax=args.optimized,
                        row_amax=args.row_scaled,
                        chunked_rows=args.chunked_rows,
                    )
                    b = packed[key]
                    if args.chunked_rows:
                        fused = pack_operand(x, row_amax=True)
                        if not (
                            torch.equal(
                                a.codes.view(torch.uint8), fused.codes.view(torch.uint8)
                            )
                            and torch.equal(
                                a.scales.view(torch.uint8),
                                fused.scales.view(torch.uint8),
                            )
                            and torch.equal(a.inverse, fused.inverse)
                        ):
                            raise ValueError("chunked row packing changed quantization")
                    if args.optimized and not args.row_scaled:
                        ordinary = pack_operand(x)
                        if not (
                            torch.equal(
                                a.codes.view(torch.uint8),
                                ordinary.codes.view(torch.uint8),
                            )
                            and torch.equal(
                                a.scales.view(torch.uint8),
                                ordinary.scales.view(torch.uint8),
                            )
                            and torch.equal(a.inverse, ordinary.inverse)
                        ):
                            raise ValueError("fused amax changed NVFP4 quantization")
                    start = time.perf_counter()
                    op = Nvfp4Gemm(m, k, n)
                    torch.cuda.synchronize()
                    compile_seconds = time.perf_counter() - start
                    y = op(a, b)
                    oracle = (decode_operand(a) @ decoded[key].t()).to(torch.bfloat16)
                    bf16 = F.linear(x, w)
                    perturbed = x.clone()
                    perturbed[0] *= 31.7
                    changed = op(
                        pack_operand(
                            perturbed,
                            fused_amax=args.optimized,
                            row_amax=args.row_scaled,
                            chunked_rows=args.chunked_rows,
                        ),
                        b,
                    )
                    isolation = relative_l2(changed[1:], y[1:])
                    if args.row_scaled and isolation != 0:
                        raise ValueError("per-row FP4 packing couples independent rows")
                    row = {
                        "tokens": m,
                        "path": key,
                        "packing_kernel_metadata": {
                            str(k): v for k, v in PACKING_KERNEL_METADATA.items()
                        },
                        "mkn": [m, k, n],
                        "compile_seconds": compile_seconds,
                        "route": op.plan.route,
                        "tile": op.plan.tile_config_name,
                        "decoded_oracle_relative_l2": relative_l2(y, oracle),
                        "bf16_relative_l2": relative_l2(y, bf16),
                        "finite": bool(torch.isfinite(y).all()),
                        "cross_row_perturbation_relative_l2": isolation,
                    }
                    if args.row_scaled and m == 193:
                        global_y = op(pack_operand(x, fused_amax=True), b)
                        global_changed = op(pack_operand(perturbed, fused_amax=True), b)
                        row["tensor_global_cross_row_relative_l2"] = relative_l2(
                            global_changed[1:], global_y[1:]
                        )
                    report["shapes"].append(row)
                    save()
                    print(json.dumps(row), flush=True)
                    if not row["finite"] or row["decoded_oracle_relative_l2"] > 0.01:
                        raise ValueError(
                            "native FP4 arithmetic failed decoded-operand gate"
                        )
                    row["timing"] = measure(
                        {
                            "bf16": partial(F.linear, x, w),
                            "fp4_prepacked": partial(op, a, b),
                            "fp4_including_pack": lambda op=op, x=x, b=b: op(
                                pack_operand(
                                    x,
                                    fused_amax=args.optimized,
                                    row_amax=args.row_scaled,
                                    chunked_rows=args.chunked_rows,
                                ),
                                b,
                            ),
                        }
                    )
                    t = row["timing"]["mean_ms"]
                    row["relative_improvement_including_pack"] = (
                        1 - t["fp4_including_pack"] / t["bf16"]
                    )
                    if args.optimized:
                        row["graph_timing"] = graph_measure(
                            x,
                            w,
                            b,
                            op,
                            row_scaled=args.row_scaled,
                            chunked_rows=args.chunked_rows,
                        )
                    row["pack_only_timing"] = measure(
                        {
                            "pack": partial(
                                pack_operand,
                                x,
                                fused_amax=args.optimized,
                                row_amax=args.row_scaled,
                                chunked_rows=args.chunked_rows,
                            )
                        }
                    )
                    save()
                    print(json.dumps(row), flush=True)
                    del x, a, y, oracle, bf16, op, perturbed, changed
        report["status"] = "complete"
    except Exception as exc:
        report.update(
            status="failed", error=repr(exc), traceback=traceback.format_exc()
        )
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
