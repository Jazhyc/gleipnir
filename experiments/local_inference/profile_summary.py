"""Summarize actual CUDA kernel durations, never CPU dispatch durations."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments.local_inference.core import write_json


def kernel_family(name: str) -> str:
    """Conservative name-based categories; ambiguous fused kernels stay other."""
    if "gemm" in name or "gemvx" in name:
        return "matrix_multiplication"
    if "flash::" in name:
        return "flash_attention"
    if any(
        part in name
        for part in (
            "chunk_",
            "conv1d",
            "_fused_post_conv",
            "merge_16x16",
            "recompute_w_u",
        )
    ):
        return "named_gdn_and_convolution"
    if any(part in name for part in ("reshape_and_cache", "kv_blocks", "slot_mapping")):
        return "kv_write_and_bookkeeping"
    return "other"


def summarize_trace(trace: dict[str, Any]) -> dict[str, Any]:
    kernels = [
        e
        for e in trace["traceEvents"]
        if e.get("cat") == "kernel" and e.get("ph") == "X"
    ]
    if not kernels:
        raise ValueError("No CUDA kernel events; CPU-only trace is not a GPU profile")
    totals = defaultdict(float)
    counts = defaultdict(int)
    families = defaultdict(float)
    for event in kernels:
        totals[event["name"]] += event["dur"]
        counts[event["name"]] += 1
        families[kernel_family(event["name"])] += event["dur"]
    total = sum(totals.values())
    intervals = sorted((e["ts"], e["ts"] + e["dur"]) for e in kernels)
    busy = 0.0
    start, end = intervals[0]
    first = start
    for a, b in intervals[1:]:
        if a > end:
            busy += end - start
            start, end = a, b
        else:
            end = max(end, b)
    busy += end - start
    span = end - first
    return {
        "kernel_count": len(kernels),
        "summed_kernel_seconds": total / 1e6,
        "first_to_last_kernel_seconds": span / 1e6,
        "kernel_active_union_seconds": busy / 1e6,
        "kernel_active_fraction": busy / span if span else None,
        "family_percent_summed_kernel_time": {
            name: duration / total * 100
            for name, duration in sorted(families.items(), key=lambda x: -x[1])
        },
        "kernels": [
            {
                "name": name,
                "calls": counts[name],
                "seconds": duration / 1e6,
                "percent_summed_kernel_time": duration / total * 100,
            }
            for name, duration in sorted(totals.items(), key=lambda x: -x[1])
        ],
        "limitations": (
            "Instrumented bounded trace, not benchmark throughput. Kernel-time shares "
            "are not compute or bandwidth utilization. Union combines streams; "
            "intended for this single-GPU workload. Gaps can include profiler overhead."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    opener = gzip.open if args.trace.suffix == ".gz" else open
    with opener(args.trace, "rt") as handle:
        result = summarize_trace(json.load(handle))
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
