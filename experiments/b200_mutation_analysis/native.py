"""Reproduce scalar metadata failure and check exact native mutation semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import types
from pathlib import Path

import torch
import triton
from vllm.model_executor.layers import fused_qk_norm_rope as module

from gleipnir.serving.triton_mutation import KERNEL, OUTPUTS, install, original_analysis

ROOT = Path(__file__).resolve().parents[2]


def inputs(rows: int, strided: bool = False) -> tuple:
    def matrix(width):
        value = torch.randn(
            rows * (2 if strided else 1), width, device="cuda", dtype=torch.bfloat16
        )
        return value[::2] if strided else value

    return (
        matrix(8192),
        matrix(1024),
        torch.randn(256, device="cuda", dtype=torch.bfloat16) * 0.1 + 1,
        torch.randn(256, device="cuda", dtype=torch.bfloat16) * 0.1 + 1,
        torch.randn(32768, 64, device="cuda", dtype=torch.bfloat16),
        torch.arange(rows, device="cuda", dtype=torch.int64),
    )


def invoke(function, args):
    return function(
        *args, eps=1e-6, num_q_heads=16, num_kv_heads=4, head_dim=256, rotary_dim=64
    )


def clone_function(function):
    """Freeze each kernel global so changing the module cannot change the control."""
    result = types.FunctionType(
        function.__code__,
        dict(function.__globals__),
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def mutation_args(args):
    q, k, qw, kw, cache, positions = args
    out_q = torch.empty(q.shape[0], 4096, dtype=q.dtype, device=q.device)
    out_k = torch.empty_like(k)
    gate = torch.empty_like(out_q)
    values = [
        q,
        k,
        out_q,
        out_k,
        gate,
        qw,
        kw,
        cache,
        positions,
        q.stride(0),
        k.stride(0),
        out_q.stride(0),
        out_k.stride(0),
        gate.stride(0),
        cache.stride(0),
        16,
        4,
        256,
        64,
        32,
        1e-6,
        triton.language.bfloat16,
        256,
        32,
        True,
    ]
    return dict(zip(getattr(module, KERNEL).arg_names, values, strict=True))


def graph_timing(function, args):
    # Exclude compilation and allocator/stream warmup; retain the graph for replay.
    for _ in range(3):
        invoke(function, args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = invoke(function, args)
    timings = []
    for _ in range(5):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        for _ in range(50):
            graph.replay()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) / 50)
    return timings, graph, output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    original_kernel = getattr(module, KERNEL)
    original = clone_function(module.fused_qk_rmsnorm_rope_gate)
    fixed_metadata = mutation_args(inputs(17))
    import torch._higher_order_ops.triton_kernel_wrap as wrap

    try:
        wrap.generate_ttir(original_kernel, dict(fixed_metadata), {})
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        if "Function argument index out of range" not in failure:
            raise
    else:
        raise ValueError("original mutation failure did not reproduce")
    record = install(ROOT)
    corrected_kernel = getattr(module, KERNEL)
    wrap.generate_ttir(corrected_kernel, dict(fixed_metadata), {})
    mutations = wrap.identify_mutated_tensors(
        corrected_kernel, dict(fixed_metadata), {}
    )
    assert set(mutations) == OUTPUTS, mutations
    fixed = clone_function(module.fused_qk_rmsnorm_rope_gate)

    # Head geometry is a model constant. Making these Python integers dynamic
    # breaks even the unmodified control's constexpr head/block parameters.
    # Keep tensor dimensions/strides dynamic, and freeze the actual model geometry.
    def legacy_forward(q, k, qw, kw, cache, positions, **unused):
        return original(
            q,
            k,
            qw,
            kw,
            cache,
            positions,
            eps=1e-6,
            num_q_heads=16,
            num_kv_heads=4,
            head_dim=256,
            rotary_dim=64,
        )

    def repaired_forward(q, k, qw, kw, cache, positions, **unused):
        return fixed(
            q,
            k,
            qw,
            kw,
            cache,
            positions,
            eps=1e-6,
            num_q_heads=16,
            num_kv_heads=4,
            head_dim=256,
            rotary_dim=64,
        )

    old_compiled = torch.compile(legacy_forward, fullgraph=True, dynamic=True)
    new_compiled = torch.compile(repaired_forward, fullgraph=True, dynamic=True)
    report = {
        "passed": False,
        "original_failure": failure,
        "corrected_mutations": mutations,
        "helper_sha256": record["helper_sha256"],
        "installation": record,
        "checks": [],
        "profiles": [],
        "gpu": torch.cuda.get_device_name(),
    }
    out.write_text(json.dumps(report, indent=2) + "\n")
    started = time.perf_counter()
    for rows, strided in [
        (1, False),
        (17, False),
        (129, False),
        (1536, False),
        (4096, False),
        (32768, False),
        (17, True),
        (4096, True),
    ]:
        x = inputs(rows, strided)
        saved = [t.clone() for t in x]
        eager = invoke(original, x)
        fixed_eager = invoke(fixed, x)
        with original_analysis():
            legacy = invoke(old_compiled, x)
        corrected = invoke(new_compiled, x)
        torch.cuda.synchronize()
        exact = all(torch.equal(a, b) for a, b in zip(eager, fixed_eager, strict=True))
        compiled_exact = all(
            torch.equal(a, b) and torch.equal(a, c)
            for a, b, c in zip(eager, legacy, corrected, strict=True)
        )
        unchanged = all(torch.equal(a, b) for a, b in zip(x, saved, strict=True))
        finite = all(torch.isfinite(y).all().item() for y in corrected)
        with original_analysis():
            old_time, old_graph, old_output = graph_timing(old_compiled, x)
        new_time, new_graph, new_output = graph_timing(new_compiled, x)
        # Mutate static graph inputs, then require replay to produce new, exact outputs.
        before = [y.clone() for y in new_output]
        x[0].normal_()
        x[1].normal_()
        old_graph.replay()
        new_graph.replay()
        torch.cuda.synchronize()
        changed_eager = invoke(original, x)
        replay_exact = all(
            torch.equal(a, b) and torch.equal(a, c)
            for a, b, c in zip(changed_eager, old_output, new_output, strict=True)
        )
        changed = any(
            not torch.equal(a, b) for a, b in zip(before, new_output, strict=True)
        )
        check = {
            "rows": rows,
            "strided": strided,
            "eager_exact": exact,
            "compiled_exact": compiled_exact,
            "inputs_unchanged": unchanged,
            "finite": finite,
            "changed_input_replay_exact": replay_exact,
            "replay_changed": changed,
            "old_graph_ms": old_time,
            "fixed_graph_ms": new_time,
        }
        report["checks"].append(check)
        out.write_text(json.dumps(report, indent=2) + "\n")
        assert (
            exact
            and compiled_exact
            and unchanged
            and finite
            and replay_exact
            and changed
        ), check
        print("mutation_native_case", rows, strided, flush=True)
    # Instrument one common shape only after native timing; no speed claim uses it.
    x = inputs(4096)
    for name, function in [("original", old_compiled), ("fixed", new_compiled)]:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile:
            invoke(function, x)
            torch.cuda.synchronize()
        target = out.with_name(f"{out.stem}_{name}.trace.json")
        profile.export_chrome_trace(str(target))
        relevant = [
            {"operator": event.key, "count": event.count}
            for event in profile.key_averages()
            if any(k in event.key for k in ["clone", "copy", "fused"])
        ]
        report["profiles"].append(
            {
                "mode": name,
                "trace": str(target),
                "operators": relevant,
                "timed_benchmark": False,
            }
        )
    report.update(passed=True, seconds=time.perf_counter() - started)
    report["executed_source_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("mutation_native_passed", report["seconds"], flush=True)


if __name__ == "__main__":
    main()
