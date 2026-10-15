"""Summarize torch-profiler traces: GPU busy/idle time and kernel categories."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

# Ordered: the first matching pattern names a kernel's category.
CATEGORIES = (
    # cuDNN DSL MXFP8 SDPA (double underscore); FP4 cuDNN kernels are named.
    ("attention_mxfp8", r"^cudnn_kernel__kernel_"),
    ("attention_pack", r"_produce|_token_offsets"),
    ("kv_cache_write", r"reshape_and_cache"),
    (
        "gdn",
        r"flashinfergdn|gated_delta|delta_rule|causal_conv1d|_fused_post_conv"
        r"|layer_norm_fwd",
    ),
    ("gemm_fp4", r"frost_sm100_block_scale|swiglu_output|BlockScaled"),
    ("fp4_quant_pack", r"nvfp4Quant|_norm_pack|_silu_pack|_row_scale|_pack\b"),
    ("gemm_fp8", r"GemmUniversal|cutlass.*[Gg]emm"),
    ("fp8_quant", r"scaled_fp8_quant"),
    ("gemm_bf16_cublas", r"nvjet|cublas|xmma"),
    ("norm_rope", r"rms_norm|rmsnorm|rope|rotary"),
    ("memory", r"Memcpy|Memset|copy|index|zeros|fill|cat_"),
    ("elementwise", r"triton_|elementwise|reduce"),
)


def categorize(name: str) -> str:
    for category, pattern in CATEGORIES:
        if re.search(pattern, name, flags=re.IGNORECASE):
            return category
    return "other"


def merged_busy(intervals: list[tuple[float, float]]) -> tuple[float, float]:
    intervals.sort()
    merged: list[list[float]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    busy = sum(b - a for a, b in merged)
    return busy, merged[-1][1] - merged[0][0]


def summarize(path: Path, top: int = 40) -> dict:
    with gzip.open(path, "rt") as handle:
        events = json.load(handle)["traceEvents"]
    kernels = [
        e
        for e in events
        if e.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}
        and e.get("dur", 0) > 0
    ]
    if not kernels:
        raise ValueError("profile contains no GPU events")
    busy, window = merged_busy([(e["ts"], e["ts"] + e["dur"]) for e in kernels])
    by_name: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
    by_category: dict[str, float] = defaultdict(float)
    for e in kernels:
        by_name[e["name"]][0] += e["dur"]
        by_name[e["name"]][1] += 1
        by_category[categorize(e["name"])] += e["dur"]
    total = sum(by_category.values())
    launches = sum(
        1
        for e in events
        if e.get("cat") in {"cuda_runtime", "cuda_driver"} and "Launch" in e["name"]
    )
    return {
        "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "gpu_events": len(kernels),
        "launch_api_calls": launches,
        "gpu_window_ms": window / 1000,
        "gpu_busy_ms": busy / 1000,
        "gpu_idle_share": 1 - busy / window,
        "kernel_time_ms": total / 1000,
        "categories_ms": {
            k: v / 1000 for k, v in sorted(by_category.items(), key=lambda x: -x[1])
        },
        "category_share": {k: v / total for k, v in by_category.items()},
        "top_kernels": [
            {
                "name": name[:200],
                "category": categorize(name),
                "total_ms": duration / 1000,
                "count": count,
                "share": duration / total,
            }
            for name, (duration, count) in sorted(
                by_name.items(), key=lambda x: -x[1][0]
            )[:top]
        ],
        "diagnostic_only": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    report = {}
    for trace in sorted(args.run.glob("*/*/*.pt.trace.json.gz")):
        case = f"{trace.parent.parent.name}_{trace.parent.name}"
        report[case] = summarize(trace)
    (args.run / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    for case, summary in report.items():
        shares = ", ".join(f"{k} {v:.1f}" for k, v in summary["categories_ms"].items())
        print(
            f"{case}: window {summary['gpu_window_ms']:.1f} ms, busy "
            f"{summary['gpu_busy_ms']:.1f} ms, idle {summary['gpu_idle_share']:.1%}; "
            f"{shares}"
        )


if __name__ == "__main__":
    main()
