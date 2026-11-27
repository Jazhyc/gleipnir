"""Attribute synchronization callers and explicit dtype-copy kernel work."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def sync_group(parents: list[dict]) -> str:
    """Use recorded operator ancestry rather than summed CPU waits as GPU cost."""
    names = [p["name"] for p in parents]
    if "ChunkGatedDeltaRuleFunction" in names:
        return "flashqla_metadata"
    if "aten::repeat_interleave" in names:
        return "sequence_id_construction"
    if "aten::copy_" in names:
        if any(n.startswith("Torch-Compiled Region:") for n in names):
            return "compiled_region_transfer"
        return "input_or_layout_transfer"
    if "aten::is_nonzero" in names:
        return "boolean_scalar_read"
    if "aten::item" in names:
        return "other_scalar_read"
    return "unattributed"


def analyze(events: list[dict]) -> dict:
    """Count API synchronizations and correlate copy kernels with CPU operands."""
    rows = sorted(
        [
            e
            for e in events
            if e.get("ph") == "X"
            and e.get("cat") in {"cpu_op", "user_annotation", "cuda_runtime"}
        ],
        key=lambda e: (e["ts"], -e.get("dur", 0)),
    )
    stacks = {}
    syncs = defaultdict(lambda: {"calls": 0, "cpu_ms": 0.0})
    for event in rows:
        stack = stacks.setdefault((event.get("pid"), event.get("tid")), [])
        end = event["ts"] + event.get("dur", 0)
        while stack and stack[-1]["ts"] + stack[-1].get("dur", 0) < end:
            stack.pop()
        if event["name"] == "cudaStreamSynchronize":
            group = syncs[sync_group(stack)]
            group["calls"] += 1
            group["cpu_ms"] += event.get("dur", 0) / 1000
        stack.append(event)
    operators = {
        e["args"]["External id"]: e
        for e in events
        if e.get("cat") == "cpu_op" and "External id" in e.get("args", {})
    }
    copies = defaultdict(lambda: {"calls": 0, "gpu_ms": 0.0})
    kernels = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]
    for event in kernels:
        if "copy_kernel" not in event["name"]:
            continue
        op = operators.get(event.get("args", {}).get("External id"), {})
        args = op.get("args", {})
        shapes = args.get("Input Dims", [])
        shape = shapes[0] if shapes else []
        key = json.dumps(
            {
                "types_destination_source": args.get("Input type", [])[:2],
                "rank": len(shape),
                "last_dimension": shape[-1:] if shape else [],
            },
            sort_keys=True,
        )
        copies[key]["calls"] += 1
        copies[key]["gpu_ms"] += event.get("dur", 0) / 1000
    return {
        "cuda_kernel_calls": len(kernels),
        "summed_kernel_ms": sum(e.get("dur", 0) for e in kernels) / 1000,
        "stream_synchronization_calls": sum(g["calls"] for g in syncs.values()),
        "synchronization_callers": dict(syncs),
        "explicit_copy_groups": [
            {**json.loads(key), **value}
            for key, value in sorted(copies.items(), key=lambda row: -row[1]["gpu_ms"])
        ],
        "limitations": [
            "CPU wait durations overlap GPU work and are not additive savings.",
            "Compiled-region transfer is ancestry evidence, not a Python stack.",
            "Explicit copies exclude casts fused into arithmetic kernels.",
            "Instrumented shares are not unprofiled component wall times.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    source = args.trace.read_bytes()
    report = analyze(json.loads(source)["traceEvents"])
    report.update(
        trace_sha256=hashlib.sha256(source).hexdigest(),
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    output = args.trace.parent / "hotpath_analysis.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
