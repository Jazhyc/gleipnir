"""Attribute an existing Chrome trace without rerunning or changing training."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def cpu_contexts(events: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Map external IDs to nested CPU operators and original coarse scopes."""
    threads: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for event in events:
        if event.get("cat") == "cpu_op" and event.get("ph") == "X":
            threads[event["pid"], event["tid"]].append(event)
    if len({pid for pid, _ in threads}) > 1:
        raise ValueError("external-ID attribution requires a single CPU process")
    contexts = {}
    for thread_events in threads.values():
        stack: list[tuple[float, dict]] = []
        for event in sorted(thread_events, key=lambda e: (e["ts"], -e["dur"])):
            start, end = event["ts"], event["ts"] + event["dur"]
            while stack and (stack[-1][0] <= start or stack[-1][0] + 0.01 < end):
                stack.pop()
            parent = stack[-1][1] if stack else {}
            name = event["name"]
            scope = parent.get("scope", "other")
            backward = parent.get("in_backward", False) or "Backward" in name
            phase = parent.get("fla_phase")
            if name in {"FrozenFp4Function", "gleipnir::frozen_fp4_forward"}:
                scope = "native_fp4_forward"
            elif name in {"FrozenFp4FunctionBackward", "gleipnir::frozen_fp4_backward"}:
                scope = "decoded_bf16_backward"
            elif name.startswith("ChunkGatedDeltaRuleFunction"):
                scope = "fla"
                phase = (
                    "backward"
                    if "Backward" in name
                    else "checkpoint_forward"
                    if parent.get("in_backward", False)
                    else "original_forward"
                )
            context = {
                "scope": scope,
                "operator": name,
                "in_backward": backward,
                "fla_phase": phase,
            }
            external_id = event.get("args", {}).get("External id")
            if external_id is not None:
                if external_id in contexts:
                    raise ValueError(f"duplicate CPU external ID: {external_id}")
                contexts[external_id] = context
            stack.append((end, context))
    return contexts


def other_family(name: str, operator: str) -> str:
    """Partition coarse 'other' activity by observable kernel/operator names."""
    lower = name.lower()
    if "flash_bwd" in lower or "fmha_cutlassb" in lower:
        return "full_attention_backward"
    if "flash_fwd" in lower or "fmha_cutlassf" in lower:
        return "full_attention_forward"
    if "causal_conv1d" in lower:
        return "causal_conv1d"
    if "dequant" in lower or "dequantize" in operator.lower():
        return "weight_dequantization"
    if any(key in lower for key in ("nvjet", "gemm", "xmma", "cublas")):
        return "other_matrix_multiplication"
    if "copy_kernel" in lower or "direct_copy" in lower:
        return "casts_and_tensor_copies"
    if any(key in lower for key in ("layer_norm", "l2norm", "rms_norm", "rmsnorm")):
        return "normalization"
    if any(key in lower for key in ("reduce_kernel", "triton_red_", "lpnorm")):
        return "reductions"
    if any(key in lower for key in ("catarray", "index", "scatter", "gather", "scan")):
        return "layout_and_indexing"
    if "triton_poi_" in lower:
        return "compiled_pointwise"
    if "elementwise_kernel" in lower:
        return "eager_pointwise"
    return "remaining_named_kernels"


def _ranked(values: dict[tuple, dict], labels: list[str]) -> list[dict]:
    return [
        {**dict(zip(labels, key, strict=True)), **value}
        for key, value in sorted(values.items(), key=lambda item: -item[1]["seconds"])
    ]


def summarize_trace(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Count GPU kernel events once; CPU scopes supply labels, never durations."""
    contexts = cpu_contexts(events)
    scopes = defaultdict(lambda: {"calls": 0, "seconds": 0.0})
    families = defaultdict(lambda: {"calls": 0, "seconds": 0.0})
    operators = defaultdict(lambda: {"calls": 0, "seconds": 0.0})
    named = defaultdict(lambda: {"calls": 0, "seconds": 0.0})
    fla_phases = defaultdict(lambda: {"calls": 0, "seconds": 0.0})
    fla_kernels = defaultdict(lambda: {"calls": 0, "seconds": 0.0})
    fp4_kernels = defaultdict(lambda: {"calls": 0, "seconds": 0.0})
    grids = defaultdict(lambda: {"calls": 0, "seconds": 0.0})
    unmapped = {"calls": 0, "seconds": 0.0}
    for event in events:
        if event.get("cat") != "kernel" or event.get("ph") != "X":
            continue
        args = event.get("args", {})
        context = contexts.get(args.get("External id"))
        if context is None:
            unmapped["calls"] += 1
            unmapped["seconds"] += event["dur"] / 1e6
            context = {"scope": "other", "operator": "unmapped"}
        scope, operator = context["scope"], context["operator"]
        name, seconds = event["name"], event["dur"] / 1e6
        targets = [scopes[(scope,)]]
        if scope == "other":
            family = other_family(name, operator)
            targets += [
                families[(family,)],
                operators[(family, operator)],
                named[(family, name)],
            ]
        elif scope == "fla":
            phase = context.get("fla_phase") or "unknown"
            targets += [fla_phases[(phase,)], fla_kernels[(phase, name)]]
            if name.startswith("chunk_gated_delta_rule_fwd_kernel_h"):
                targets.append(grids[(phase, tuple(args.get("grid", [])))])
        elif scope == "native_fp4_forward":
            targets.append(fp4_kernels[(name,)])
        for value in targets:
            value["calls"] += 1
            value["seconds"] += seconds
    return {
        "kernel_scopes": _ranked(scopes, ["scope"]),
        "other_families": _ranked(families, ["family"]),
        "other_operators": _ranked(operators, ["family", "operator"]),
        "other_kernels": _ranked(named, ["family", "kernel"]),
        "fla_phases": _ranked(fla_phases, ["phase"]),
        "fla_kernels": _ranked(fla_kernels, ["phase", "kernel"]),
        "fla_state_forward_grids": _ranked(grids, ["phase", "grid"]),
        "fp4_forward_kernels": _ranked(fp4_kernels, ["kernel"]),
        "unmapped_kernels": unmapped,
        "limits": (
            "One instrumented batch, no optimizer update. Summed GPU kernel "
            "durations can overlap and include profiling overhead. Names identify "
            "operation families, not model modules. Shapes and Python stacks were "
            "not recorded. Checkpoint-forward means a FLA forward nested inside "
            "a CPU backward scope. No throughput or correctness claim for new kernels."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    trace_path = args.root / "fouroversix/profile_trace.json"
    result = summarize_trace(json.loads(trace_path.read_text())["traceEvents"])
    with trace_path.open("rb") as stream:
        result["trace_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    result["analysis_source_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    coarse = json.loads((args.root / "profile_analysis.json").read_text())
    if {scope["scope"] for scope in result["kernel_scopes"]} != set(
        coarse["kernel_scopes"]
    ):
        raise ValueError("coarse profile scope set mismatch")
    for scope in result["kernel_scopes"]:
        expected = coarse["kernel_scopes"][scope["scope"]]
        if (
            scope["calls"] != expected["calls"]
            or abs(scope["seconds"] - expected["seconds"]) > 1e-8
        ):
            raise ValueError(f"coarse profile mismatch: {scope['scope']}")
    result["coarse_scope_reconciliation_passed"] = True
    (args.root / "profile_breakdown.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    for field in ("other_families", "fla_phases", "unmapped_kernels"):
        print(field, json.dumps(result[field]))


if __name__ == "__main__":
    main()
