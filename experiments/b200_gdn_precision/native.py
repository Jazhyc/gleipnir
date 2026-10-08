"""Validate native FP8 GDN projection geometry, isolation and graph replay."""

from __future__ import annotations

import argparse
import importlib.metadata
import statistics
import time
from pathlib import Path

import torch
from vllm import _custom_ops as ops

from experiments.b200_inference_benchmark.run import ROOT, sha, write
from gleipnir.serving.gdn_precision import ROWS, SHAPES
from gleipnir.serving.vllm.attention_precision import bf16_linear
from gleipnir.serving.vllm.gdn_precision import gdn_fp8_linear as fp8_linear


def check(precision: str, projection: str, rows: int) -> dict:
    k, n = SHAPES[projection]
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
    x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
    x[0].zero_()
    if precision == "fp8":
        codes, scales = ops.scaled_fp8_quant(
            weight, scale=None, use_per_token_if_dynamic=True
        )
        packed = codes.t()

        def forward():
            return fp8_linear(x, packed, scales)

        decoded = packed[:, :16].float() * scales[:16].reshape(1, -1)
    else:

        def forward():
            return bf16_linear(x, weight)

        decoded = weight[:16].float().t()

    def reference():
        activation = x.float()
        if precision == "fp8":
            a, s = ops.scaled_fp8_quant(x, scale=None, use_per_token_if_dynamic=True)
            activation = a[:rows].float() * s[:rows]
        return (activation @ decoded).to(torch.bfloat16)

    for _ in range(3):
        forward()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = forward()
    graph.replay()
    torch.cuda.synchronize()
    original = captured.clone()
    expected = reference()
    bf16_reference = (x.float() @ weight[:16].float().t()).to(torch.bfloat16)

    def relative(a, b):
        return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))

    error = relative(original[:, :16], expected)
    precision_error = relative(original[:, :16], bf16_reference)
    zero = bool((original[0] == 0).all())
    x[0].fill_(0.125)
    graph.replay()
    torch.cuda.synchronize()
    replay_error = relative(captured[:, :16], reference())
    untouched = torch.equal(original[1:], captured[1:])
    changed = not torch.equal(original[0], captured[0])
    finite = bool(torch.isfinite(captured).all())
    timings = []
    for _ in range(5):
        start, stop = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        for _ in range(10):
            graph.replay()
        stop.record()
        stop.synchronize()
        timings.append(start.elapsed_time(stop) / 10)
    receipt = {
        "precision": precision,
        "projection": projection,
        "rows": rows,
        "shape": [k, n],
        "finite": finite,
        "zero_row_exact": zero,
        "unchanged_rows_exact": untouched,
        "replay_changed": changed,
        "relative_l2": error,
        "replay_relative_l2": replay_error,
        "bf16_relative_l2": precision_error,
        "median_ms": statistics.median(timings),
        "weight_packing_excluded": True,
        "activation_packing_included": True,
    }
    receipt["passed"] = (
        finite and zero and untouched and changed and max(error, replay_error) <= 0.01
    )
    if not receipt["passed"]:
        raise ValueError(f"projection native validation failed: {receipt}")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    if Path(args.name).name != args.name:
        raise ValueError("native run name must be a stem")
    out = ROOT / "results/b200_gdn_precision" / f"{args.name}.json"
    if out.exists():
        raise ValueError("native receipt already exists")
    torch.manual_seed(81031)
    torch.backends.cuda.matmul.allow_tf32 = False
    receipt = {
        "state": "running",
        "passed": False,
        "gpu": torch.cuda.get_device_name(),
        "checks": [],
        "runtime": {
            n: importlib.metadata.version(n) for n in ("vllm", "torch", "triton")
        },
        "sources": {
            p: sha(ROOT / p)
            for p in [
                "src/gleipnir/serving/vllm/attention_precision.py",
                "src/gleipnir/serving/vllm/gdn_precision.py",
                "src/gleipnir/serving/gdn_precision.py",
                "src/gleipnir/serving/vllm/frost_gdn.py",
                "experiments/b200_gdn_precision/native.py",
            ]
        },
    }
    started = time.perf_counter()
    try:
        for precision in ("fp8",):
            for projection in SHAPES:
                for rows in sorted(ROWS):
                    value = check(precision, projection, rows)
                    receipt["checks"].append(value)
                    write(out, receipt)
                    print(
                        "native_case",
                        precision,
                        projection,
                        rows,
                        value["relative_l2"],
                        flush=True,
                    )
        receipt.update(
            state="completed", passed=True, seconds=time.perf_counter() - started
        )
        write(out, receipt)
    except BaseException as error:
        receipt.update(state="failed", error=f"{type(error).__name__}: {error}")
        write(out, receipt)
        raise


if __name__ == "__main__":
    main()
