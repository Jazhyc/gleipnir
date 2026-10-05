"""Attribute ordinary GEMMs using recorded operand shapes and CPU/GPU IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from experiments.b200_mlp_gemm.analyze_full_profile import kernel_group


def module_group(name: str) -> str:
    if "lora_" in name:
        return "lora_adapters"
    if ".linear_attn." in name:
        return "gdn_frozen_projections"
    if ".self_attn." in name:
        return "full_attention_frozen_projections"
    if "lm_head" in name:
        return "lm_head"
    return "other_linear"


def shape_candidates(shapes: list, modules: list[dict]) -> list[str]:
    """Return all compatible components; preserve ambiguity instead of guessing."""
    if len(shapes) < 2 or any(len(x) != 2 for x in shapes[:2]):
        return []
    (m, k), (kb, n) = shapes[:2]
    if k != kb:
        return []
    groups = set()
    for module in modules:
        out, width = module["shape"]
        if (k, n) in {(width, out), (out, width)} or (
            module["requires_grad"] and (m, n) == (out, width)
        ):
            groups.add(module_group(module["name"]))
    return sorted(groups)


def operator_context(events: list[dict]) -> tuple[dict, dict]:
    """Identify decoder regions from actual attention calls and link autograd nodes."""
    rows = sorted(
        [
            e
            for e in events
            if e.get("ph") == "X"
            and "dur" in e
            and e.get("cat") in {"cpu_op", "user_annotation"}
        ],
        key=lambda e: (e["ts"], -e["dur"]),
    )
    stacks, ancestry, region_kinds = {}, {}, defaultdict(set)
    markers = {
        "ChunkGatedDeltaRuleFunction": "gdn",
        "FlashAttnVarlenFunc": "full_attention",
    }
    for event in rows:
        stack = stacks.setdefault((event.get("pid"), event.get("tid")), [])
        while stack and (
            stack[-1]["ts"] + stack[-1]["dur"] <= event["ts"]
            or stack[-1]["ts"] + stack[-1]["dur"] < event["ts"] + event["dur"]
        ):
            stack.pop()
        external = event.get("args", {}).get("External id")
        if external is not None:
            ancestry[external] = stack.copy()
        if event["name"] in markers:
            for parent in stack:
                if parent["name"].startswith("Torch-Compiled Region:"):
                    region_kinds[parent["name"]].add(markers[event["name"]])
        stack.append(event)

    def kinds(parents):
        return set().union(*(region_kinds.get(e["name"], set()) for e in parents))

    sequences = defaultdict(set)
    for event in rows:
        sequence = event.get("args", {}).get("Sequence number")
        if sequence is not None and "Backward" not in event["name"]:
            context = kinds(ancestry.get(event.get("args", {}).get("External id"), []))
            sequences[(event.get("pid"), sequence)].update(context)
    contexts = {}
    for external, parents in ancestry.items():
        context = kinds(parents)
        for parent in parents:
            if "Backward" in parent["name"]:
                context |= sequences.get(
                    (parent.get("pid"), parent.get("args", {}).get("Sequence number")),
                    set(),
                )
        if len(context) == 1:
            contexts[external] = next(iter(context))
    return contexts, {name: sorted(value) for name, value in region_kinds.items()}


def analyze(events: list[dict], modules: list[dict]) -> dict:
    contexts, region_evidence = operator_context(events)
    operators = {
        e["args"]["External id"]: e
        for e in events
        if e.get("ph") == "X"
        and e.get("cat") == "cpu_op"
        and e.get("name") in {"aten::mm", "aten::addmm", "aten::bmm"}
        and "External id" in e.get("args", {})
    }
    groups = defaultdict(lambda: {"calls": 0, "gpu_ms": 0.0})
    geometries = defaultdict(lambda: {"calls": 0, "gpu_ms": 0.0})
    total_gpu = ordinary_gpu = 0.0
    for event in events:
        if event.get("cat") != "kernel" or event.get("ph") != "X":
            continue
        ms = event["dur"] / 1000
        total_gpu += ms
        if kernel_group(event["name"]) != "ordinary_gemms_lora_and_other_projections":
            continue
        ordinary_gpu += ms
        operator = operators.get(event.get("args", {}).get("External id"))
        shapes = operator.get("args", {}).get("Input Dims", []) if operator else []
        if operator and operator["name"] == "aten::addmm":
            shapes = shapes[1:]
        candidates = shape_candidates(shapes, modules)
        context = contexts.get(event.get("args", {}).get("External id"))
        if len(candidates) > 1 and context in {"gdn", "full_attention"}:
            # LoRA weight gradients cannot be identified by decoder context alone.
            compatible = {
                "gdn": "gdn_frozen_projections",
                "full_attention": "full_attention_frozen_projections",
            }[context]
            if set(candidates) <= {
                "gdn_frozen_projections",
                "full_attention_frozen_projections",
            }:
                candidates = [compatible] if compatible in candidates else candidates
        group = (
            candidates[0]
            if len(candidates) == 1
            else ("ambiguous:" + "+".join(candidates) if candidates else "unattributed")
        )
        geometry = json.dumps(
            {
                "operator": operator["name"] if operator else None,
                "shapes": shapes,
                "candidates": candidates,
                "decoder_context": context,
            },
            sort_keys=True,
        )
        for row in (groups[group], geometries[geometry]):
            row["calls"] += 1
            row["gpu_ms"] += ms
    if ordinary_gpu <= 0:
        raise ValueError("trace has no identifiable ordinary GEMMs")
    for row in groups.values():
        row.update(
            fraction_of_ordinary_gemm_time=row["gpu_ms"] / ordinary_gpu,
            fraction_of_total_kernel_time=row["gpu_ms"] / total_gpu,
        )
    return {
        "summed_kernel_ms": total_gpu,
        "ordinary_gemm_ms": ordinary_gpu,
        "groups": dict(groups),
        "geometries": dict(geometries),
        "decoder_context_evidence": region_evidence,
        "limits": "One instrumented update. Shapes identify compatible modules; "
        "actual attention calls identify compiled decoder contexts, and autograd "
        "sequence numbers link backward contexts. Remaining ambiguity is retained. "
        "CPU/GPU External IDs link actual kernel time, without adding CPU durations.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    args = parser.parse_args()
    trace = args.session / "03gemmprofile/gemm_trace.json"
    modules = args.session / "linear_modules.json"
    report = analyze(
        json.loads(trace.read_text())["traceEvents"],
        json.loads(modules.read_text())["modules"],
    )
    report.update(
        trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
        module_metadata_sha256=hashlib.sha256(modules.read_bytes()).hexdigest(),
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (trace.parent / "gemm_analysis.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps({"groups": report["groups"]}, indent=2))


if __name__ == "__main__":
    main()
