"""Kernel-only attribution and interval accounting for warmed FP4 traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def kernel_group(name: str) -> str:
    """Only assign identifiable work; never label all ordinary GEMMs as MLPs."""
    lower = name.lower()
    if "frost" in lower and ("float4" in lower or "block_scale_matmul" in lower):
        return "mlp_frozen_fp4_gemm_and_fused_descale"
    if name in {"_pack_row_blocks", "_row_inverse", "_row_scale"}:
        return "mlp_fp4_dynamic_conversion"
    if any(
        marker in lower
        for marker in (
            "gdr",
            "tilelang_prepare_h",
            "tilelang_kkt_solve",
            "causal_conv1d",
        )
    ):
        return "gdn_scan_and_convolution"
    if "l2norm" in lower or "layer_norm_gated" in lower:
        return "gdn_normalization"
    if any(
        marker in lower
        for marker in ("flashattention", "flash_attn", "flash_attention")
    ):
        return "fa4_full_attention"
    if "gemm" in lower or lower.startswith("nvjet_"):
        return "ordinary_gemms_lora_and_other_projections"
    if "silu" in lower:
        return "silu_and_fused_pointwise_without_module_attribution"
    if "copy_kernel" in lower:
        return "tensor_copies_and_casts"
    return "other_or_unclassified"


def interval_union(intervals: list[tuple[float, float]]) -> float:
    """Return microseconds occupied without double counting overlapping events."""
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged)


def analyze(events: list[dict]) -> dict:
    groups = defaultdict(lambda: {"calls": 0, "gpu_ms": 0.0})
    kernels = defaultdict(lambda: {"calls": 0, "gpu_ms": 0.0})
    apis = defaultdict(lambda: {"calls": 0, "cpu_ms": 0.0})
    intervals, device_intervals, ranges = [], [], []
    transfers = defaultdict(lambda: {"calls": 0, "gpu_ms": 0.0})
    for event in events:
        if event.get("ph") != "X" or "dur" not in event:
            continue
        name, category = event["name"], event.get("cat")
        interval = (event["ts"], event["ts"] + event["dur"])
        if category == "kernel":
            for row in (groups[kernel_group(name)], kernels[name]):
                row["calls"] += 1
                row["gpu_ms"] += event["dur"] / 1000
            intervals.append(interval)
            device_intervals.append(interval)
        elif category in {"gpu_memcpy", "gpu_memset"}:
            row = transfers[category]
            row["calls"] += 1
            row["gpu_ms"] += event["dur"] / 1000
            device_intervals.append(interval)
        elif category in {"cuda_runtime", "cuda_driver"}:
            apis[name]["calls"] += 1
            apis[name]["cpu_ms"] += event["dur"] / 1000
        elif category == "user_annotation" and name.startswith(
            "warmed_optimizer_update_"
        ):
            ranges.append(interval)
    if not intervals:
        raise ValueError("no CUDA kernel events")
    kernel_sum = sum(row["gpu_ms"] for row in groups.values())
    for row in groups.values():
        row["fraction_of_summed_kernel_time"] = row["gpu_ms"] / kernel_sum
    span = (
        max(end for _, end in device_intervals)
        - min(start for start, _ in device_intervals)
    ) / 1000
    busy = interval_union(device_intervals) / 1000
    return {
        "summed_kernel_ms": kernel_sum,
        "kernel_count": sum(row["calls"] for row in groups.values()),
        "kernel_interval_union_ms": interval_union(intervals) / 1000,
        "device_interval_union_ms": busy,
        "first_to_last_device_span_ms": span,
        "no_device_event_ms_within_span": span - busy,
        "profiled_cpu_update_range_ms": interval_union(ranges) / 1000,
        "groups": dict(groups),
        "kernels": dict(kernels),
        "device_transfers": dict(transfers),
        "cuda_apis": dict(apis),
        "limits": (
            "Instrumented warmed update, no baseline profile. GPU annotation and CPU "
            "events are excluded from kernel sums. CPU launch/wait durations overlap "
            "GPU execution. Idle gaps do not by themselves prove a CPU bottleneck. "
            "Ordinary GEMMs and unidentified pointwise work lack module attribution."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile_directory", type=Path)
    args = parser.parse_args()
    reports = []
    for step in (11, 15, 20):
        path = args.profile_directory / f"update{step}" / "trace.json"
        report = analyze(json.loads(path.read_text())["traceEvents"])
        report.update(
            update=step, trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )
        (path.parent / "kernel_analysis.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        reports.append(report)
    combined_groups = defaultdict(lambda: {"calls": 0, "gpu_ms": 0.0})
    for report in reports:
        for group, row in report["groups"].items():
            combined_groups[group]["calls"] += row["calls"]
            combined_groups[group]["gpu_ms"] += row["gpu_ms"]
    total = sum(row["gpu_ms"] for row in combined_groups.values())
    for row in combined_groups.values():
        row["fraction_of_summed_kernel_time"] = row["gpu_ms"] / total
    summary = {
        "updates": [
            {k: v for k, v in report.items() if k != "kernels"} for report in reports
        ],
        "pooled_groups": dict(combined_groups),
        "analysis_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "limits": reports[0]["limits"],
    }
    (args.profile_directory / "analysis.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps({"pooled_groups": summary["pooled_groups"]}, indent=2))


if __name__ == "__main__":
    main()
