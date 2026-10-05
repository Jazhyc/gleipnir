"""Bounded warmed attention timing and GPU attribution; no model updates."""

from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import json
import os
import statistics
import time
from itertools import accumulate
from pathlib import Path
from unittest.mock import patch

import torch

from gleipnir.nvidia_mxfp8_varlen_attention import packed_attention

SHAPES = {
    "short_pack": (301, 280, 250, 230, 201),
    "balanced_pack": (4096, 4096, 4096, 4096),
    "skewed_pack": (14373, 1000, 992),
    "long_singleton": (24521,),
}


def attribute_kernels(events: list[dict]) -> dict:
    """Count actual kernels once, using nested CPU launch scopes as labels."""
    from collections import defaultdict

    threads = defaultdict(list)
    for event in events:
        if event.get("ph") == "X" and event.get("cat") in {
            "cpu_op",
            "user_annotation",
            "cuda_runtime",
            "cuda_driver",
        }:
            threads[event["pid"], event["tid"]].append(event)
    if len({pid for pid, _ in threads}) > 1:
        raise ValueError("external IDs require one CPU process")
    contexts = {}
    correlations = {}
    for items in threads.values():
        stack = []
        for event in sorted(items, key=lambda e: (e["ts"], -e["dur"])):
            start, end = event["ts"], event["ts"] + event["dur"]
            while stack and (stack[-1][0] <= start or stack[-1][0] + 0.01 < end):
                stack.pop()
            scope = stack[-1][1] if stack else "other"
            if event["name"].startswith("mxfp8::"):
                scope = event["name"]
            if event["cat"] in {"cuda_runtime", "cuda_driver"}:
                correlation = event.get("args", {}).get("correlation")
                if correlation is not None:
                    correlations[correlation] = scope
                continue
            external = event.get("args", {}).get("External id")
            if external is not None:
                if external in contexts:
                    raise ValueError("duplicate CPU external ID")
                contexts[external] = scope
            stack.append((end, scope))
    kernels = []
    stages = defaultdict(lambda: {"calls": 0, "us": 0.0})
    for event in events:
        if event.get("ph") != "X" or event.get("cat") != "kernel":
            continue
        args = event.get("args", {})
        scope = correlations.get(
            args.get("correlation"),
            contexts.get(args.get("External id"), "unmapped"),
        )
        kernels.append({"name": event["name"], "scope": scope, "us": event["dur"]})
        stages[scope]["calls"] += 1
        stages[scope]["us"] += event["dur"]
    return {"kernels": kernels, "gpu_kernel_stages": dict(stages)}


def analyze_existing(output: Path) -> None:
    """Reattribute saved traces, preserving the original timing receipt."""
    receipt = output / "profile_summary.json"
    original = json.loads(receipt.read_text())
    report = {
        "status": original["status"],
        "rows": [],
        "timing_receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
        "analysis_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "excludes_gpu_user_annotation": True,
    }
    for row in original["rows"]:
        trace = output / f"{row['shape']}_{row['backend']}_trace.json"
        report["rows"].append(
            {
                **{k: v for k, v in row.items() if k not in {"kernels", "operators"}},
                **attribute_kernels(json.loads(trace.read_text())["traceEvents"]),
                "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
            }
        )
    (output / "kernel_attribution.json").write_text(json.dumps(report, indent=2) + "\n")


def scoped(name, function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        with torch.profiler.record_function(name):
            return function(*args, **kwargs)

    return wrapped


@contextlib.contextmanager
def attribution():
    """Label host launch scopes without changing native kernels or cache keys."""
    import gleipnir.nvidia_mxfp8_varlen_attention as attention
    import gleipnir.nvidia_mxfp8_varlen_quantize as quantizer
    import gleipnir.nvidia_mxfp8_varlen_repack as repacker

    original_forward, original_backward = (
        attention.forward_plan,
        attention.backward_plan,
    )

    class Forward:
        def __init__(self, plan):
            self.plan = plan

        def scratch_workspace_bytes(self):
            return self.plan.scratch_workspace_bytes()

        def execute(self, *args, **kwargs):
            with torch.profiler.record_function("mxfp8::native_forward"):
                return self.plan.execute(*args, **kwargs)

    def backward(*args):
        owner, function = original_backward(*args)
        return owner, scoped("mxfp8::native_backward", function)

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            patch.object(
                quantizer, "quantize", scoped("mxfp8::quantize", quantizer.quantize)
            )
        )
        for name in ("backward_scales", "singleton_forward", "singleton_backward"):
            stack.enter_context(
                patch.object(
                    repacker, name, scoped("mxfp8::" + name, getattr(repacker, name))
                )
            )
        stack.enter_context(
            patch.object(
                attention, "forward_plan", lambda *a: Forward(original_forward(*a))
            )
        )
        stack.enter_context(patch.object(attention, "backward_plan", backward))
        yield


def timing(action, repetitions=10):
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
    torch.cuda.synchronize()
    start = time.perf_counter()
    for a, b in zip(starts, ends, strict=True):
        a.record()
        action()
        b.record()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - start) * 1000 / repetitions
    samples = [a.elapsed_time(b) for a, b in zip(starts, ends, strict=True)]
    return {
        "wall_ms": wall,
        "cuda_event_ms": statistics.mean(samples),
        "samples_ms": samples,
    }


def graph_timing(function, operands, cuts, maximum, grad):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        leaves = [x.detach().clone().requires_grad_() for x in operands]
        gradient = grad.clone()

        def action():
            out = function(*leaves, cuts, maximum)
            return torch.autograd.grad(out, leaves, gradient)

        for _ in range(2):
            action()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        action()
    result = timing(graph.replay, 20)
    del graph
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--analyze-existing", action="store_true")
    args = parser.parse_args()
    if args.analyze_existing:
        analyze_existing(args.output)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    from flash_attn.cute import flash_attn_varlen_func

    def fa4(q, k, v, cuts, maximum):
        result = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cuts,
            cu_seqlens_k=cuts,
            max_seqlen_q=maximum,
            max_seqlen_k=maximum,
            causal=True,
            softmax_scale=0.0625,
        )
        return result[0] if isinstance(result, tuple) else result

    report = {
        "status": "started",
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "rows": [],
        "native_validation_reference": (
            "results/b200_nvidia_mxfp8_varlen/native05/kernel_canary.json"
        ),
        "cache_paths": {k: v for k, v in os.environ.items() if "CACHE" in k},
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                *Path("src/gleipnir").glob("nvidia_mxfp8_varlen*.py"),
            ]
        },
        "limits": (
            "Synthetic BF16 operands, attention only; no projections, GDN, MLP "
            "or optimizer. Event timing includes launch gaps. Graph replay "
            "removes most host dispatch. Profiler overhead excluded from timings. "
            "Controlled mixed lengths are not actual training packs."
        ),
    }

    def save():
        (args.output / "profile_summary.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )

    save()
    torch.manual_seed(17)
    for index, (name, lengths) in enumerate(SHAPES.items()):
        total, maximum = sum(lengths), max(lengths)
        operands = [
            torch.randn(
                total, h, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
            )
            for h in (16, 4, 4)
        ]
        grad = torch.randn_like(operands[0])
        cuts = torch.tensor((0, *accumulate(lengths)), device="cuda", dtype=torch.int32)
        backends = [("fa4", fa4), ("mxfp8", packed_attention)]
        if index % 2:
            backends.reverse()
        for backend, function in backends:

            def action(
                function=function,
                operands=operands,
                cuts=cuts,
                maximum=maximum,
                grad=grad,
            ):
                out = function(*operands, cuts, maximum)
                gradients = torch.autograd.grad(out, operands, grad)
                return out, gradients

            print(f"warming {name} {backend}", flush=True)
            for _ in range(6):
                out, gradients = action()
            if not bool(
                torch.isfinite(out).all()
                and all(torch.isfinite(x).all() for x in gradients)
            ):
                raise RuntimeError("nonfinite diagnostic output/gradient")
            row = {
                "shape": name,
                "lengths": lengths,
                "backend": backend,
                "uncaptured": timing(action),
                "graph_replay": graph_timing(function, operands, cuts, maximum, grad),
            }
            with (
                attribution(),
                torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as profiler,
            ):
                action()
                torch.cuda.synchronize()
            trace = args.output / f"{name}_{backend}_trace.json"
            profiler.export_chrome_trace(str(trace))
            row["operators"] = [
                {
                    "name": e.key,
                    "calls": e.count,
                    "self_cpu_us": e.self_cpu_time_total,
                    "device_us": e.device_time_total,
                }
                for e in profiler.key_averages()
            ]
            row.update(attribute_kernels(json.loads(trace.read_text())["traceEvents"]))
            report["rows"].append(row)
            save()
            print(
                json.dumps(
                    {k: v for k, v in row.items() if k not in {"operators", "kernels"}}
                ),
                flush=True,
            )
        del operands, grad, cuts, out, gradients
    report["status"] = "complete"
    save()


if __name__ == "__main__":
    main()
