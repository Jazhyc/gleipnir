"""Summarize actual CUDA kernel events from a bounded instrumented trace."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

from experiments.local_inference.core import write_json
from gleipnir.qwen35_adapter_rebase import sha256_file


def category(name: str) -> str:
    """Classify names in priority order; template type names can contain CUTLASS."""
    name = name.lower()
    if name.startswith("triton_"):
        return "fused_elementwise_reduction"
    if any(t in name for t in ("flash_fwd", "flashinfer", "fmha", "attention")):
        return "attention"
    if any(
        t in name
        for t in (
            "chunk_",
            "gated_delta",
            "solve",
            "inverse",
            "recompute_w_u",
            "causal_conv",
            "post_conv",
        )
    ):
        return "gdn_and_conv"
    if any(t in name for t in ("cutlass", "gemm", "gemvx", "gemv", "matmul")):
        return "linear_gemm"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    totals, names, counts = Counter(), Counter(), Counter()
    files = list(args.directory.glob("*.pt.trace.json.gz"))
    for path in files:
        with gzip.open(path, "rt") as handle:
            trace = json.load(handle)
        for event in trace["traceEvents"]:
            if event.get("cat") != "kernel":
                continue
            name, duration = event["name"], event.get("dur", 0)
            if duration < 0:
                raise ValueError("Negative CUDA event duration")
            totals[category(name)] += duration
            names[name] += duration
            counts[name] += 1
    if not sum(counts.values()) or not sum(totals.values()):
        raise ValueError(
            "No positive-duration CUDA kernels; CPU events cannot substitute"
        )
    record = {
        "source_sha256": sha256_file(Path(__file__)),
        "trace_sha256": {str(p): sha256_file(p) for p in files},
        "kernel_events": sum(counts.values()),
        "duration_sum_us": sum(totals.values()),
        "categories_duration_us": dict(totals),
        "categories_percent": {
            k: v / sum(totals.values()) * 100 for k, v in totals.items()
        },
        "top_kernels": [
            {"name": n, "duration_us": v, "calls": counts[n]}
            for n, v in names.most_common(30)
        ],
        "note": "Heuristic name grouping; overlapping kernel duration sums "
        "are not elapsed serving time. Instrumented partial workload only.",
    }
    write_json(args.directory / "kernel_summary.json", record)
    print(
        json.dumps(
            {"events": record["kernel_events"], "percent": record["categories_percent"]}
        )
    )


if __name__ == "__main__":
    main()
