"""Count actual CUDA kernels once; retain opaque/unclassified operation time."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def kernel_group(name: str) -> str:
    if "BlackwellFmhaBackwardDQ256" in name:
        return "attention_dq_and_dot"
    if "BlackwellFmhaBackwardDKDV256" in name:
        return "attention_dkdv"
    if name.startswith("cudnn_kernel__kernel_TensorMap") and "Float8E4M3FN" in name:
        return "attention_forward"
    if name in {"_prepare", "norm_rope_backward"}:
        return "attention_operand_producer_and_norm_backward"
    if "gdr" in name or "tilelang_prepare_h" in name or "causal_conv1d" in name:
        return "gdn_and_convolution"
    if name.startswith("nvjet_") or "gemm" in name.lower():
        return "gemms_not_attributed_to_model_components"
    if "direct_copy_kernel_cuda" in name:
        return "tensor_copy_and_conversion"
    return "other_or_unclassified"


def analyze(events: list[dict]) -> dict:
    groups = defaultdict(lambda: {"calls": 0, "gpu_ms": 0.0})
    kernels = defaultdict(lambda: {"calls": 0, "gpu_ms": 0.0})
    intervals = []
    for e in events:
        if e.get("ph") != "X" or e.get("cat") != "kernel":
            continue
        name = e["name"]
        duration = e["dur"] / 1000
        for row in (groups[kernel_group(name)], kernels[name]):
            row["calls"] += 1
            row["gpu_ms"] += duration
        intervals.append((e["ts"], e["ts"] + e["dur"]))
    if not intervals:
        raise ValueError("no CUDA kernel events")
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    total = sum(r["gpu_ms"] for r in groups.values())
    for row in groups.values():
        row["fraction_of_summed_kernel_time"] = row["gpu_ms"] / total
    return {
        "summed_kernel_ms": total,
        "kernel_count": sum(r["calls"] for r in groups.values()),
        "kernel_interval_union_ms": sum(end - start for start, end in merged) / 1000,
        "first_to_last_kernel_span_ms": (merged[-1][1] - merged[0][0]) / 1000,
        "groups": dict(groups),
        "kernels": dict(kernels),
        "limits": (
            "One profiled warmup update. Summed kernel time is not wall time; "
            "GEMMs mix base, LoRA, projection and MLP operations. "
            "No module-specific GEMM attribution or baseline causal diagnosis "
            "is claimed."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    report = analyze(json.loads(args.trace.read_text())["traceEvents"])
    report.update(
        trace_sha256=hashlib.sha256(args.trace.read_bytes()).hexdigest(),
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    path = args.trace.with_name("kernel_analysis.json")
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "kernels"}, indent=2))


if __name__ == "__main__":
    main()
