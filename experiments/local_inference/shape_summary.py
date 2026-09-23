"""Correlate GPU GEMM duration with recorded aten::mm matrix dimensions."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments.local_inference.core import write_json


def projection_name(k: int, n: int) -> str:
    """Frozen Gleipnir 4B dimensions, verified against weights and vLLM fusion."""
    return {
        (2560, 18432): "mlp.gate_up_proj",
        (9216, 2560): "mlp.down_proj",
        (2560, 12288): "linear_attn.in_proj_qkvz",
        (2560, 64): "linear_attn.in_proj_ba",
        (2560, 10240): "self_attn.qkv_proj (includes query gate)",
        (4096, 2560): "attention output (linear_attn.out_proj or self_attn.o_proj)",
        (2560, 248320): "lm_head",
    }.get((k, n), "unmapped")


def summarize_shapes(events: list[dict[str, Any]]) -> dict[str, Any]:
    operators = {}
    for event in events:
        if event.get("cat") == "cpu_op" and event.get("name") == "aten::mm":
            args = event.get("args", {})
            if "External id" in args:
                operators[args["External id"]] = args.get("Input Dims", [])
    groups: dict[tuple, dict] = {}
    gpu_total = matched_total = 0.0
    unmatched = defaultdict(float)
    for event in events:
        if event.get("cat") != "kernel" or event.get("ph") != "X":
            continue
        gpu_total += event["dur"]
        dims = operators.get(event.get("args", {}).get("External id"))
        if not dims or len(dims) < 2 or any(len(x) != 2 for x in dims[:2]):
            if "gemm" in event["name"] or "gemvx" in event["name"]:
                unmatched[event["name"]] += event["dur"]
            continue
        (m, k), (k2, n) = dims[:2]
        if k != k2:
            raise ValueError("Invalid GEMM inner dimensions")
        key = (m, n, k, event["name"])
        group = groups.setdefault(
            key,
            {
                "m": m,
                "n": n,
                "k": k,
                "kernel": event["name"],
                "projection": projection_name(k, n),
                "calls": 0,
                "gpu_us": 0.0,
            },
        )
        group["calls"] += 1
        group["gpu_us"] += event["dur"]
        matched_total += event["dur"]
    if not groups:
        raise ValueError("No GPU kernels correlated with recorded GEMM shapes")
    for group in groups.values():
        group["percent_all_kernel_time"] = group["gpu_us"] / gpu_total * 100
        group["mean_us"] = group["gpu_us"] / group["calls"]
    return {
        "all_kernel_seconds": gpu_total / 1e6,
        "matched_mm_seconds": matched_total / 1e6,
        "unmatched_gemm_seconds": sum(unmatched.values()) / 1e6,
        "unmatched_gemm_names": dict(unmatched),
        "shapes": sorted(groups.values(), key=lambda x: -x["gpu_us"]),
        "limitations": "Instrumented sample; dimension mapping is not a module hook. "
        "Attention output projections share dimensions and remain grouped.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with gzip.open(args.trace, "rt") as handle:
        report = summarize_shapes(json.load(handle)["traceEvents"])
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
