"""Matched warmed attention diagnostic for fused MXFP8 preparation."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
from itertools import accumulate
from pathlib import Path

import torch

from experiments.b200_nvidia_mxfp8_varlen.profile_attention import (
    SHAPES,
    attribute_kernels,
    graph_timing,
    timing,
)
from gleipnir.nvidia_mxfp8_fused_attention import packed_attention as fused_attention
from gleipnir.nvidia_mxfp8_varlen_attention import packed_attention as old_attention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
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
        "status": "starting",
        "rows": [],
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "cache_paths": {k: v for k, v in os.environ.items() if "CACHE" in k},
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                *Path("src/gleipnir").glob("nvidia_mxfp8_fused*.py"),
            ]
        },
        "limits": (
            "Synthetic operands, attention-only; no model updates. Uncaptured "
            "and graph timing exclude profiler overhead. No whole-model time "
            "share or quality claim."
        ),
    }
    for name in report["source_sha256"]:
        path = Path(name)
        rel = path.relative_to(Path.cwd()) if path.is_absolute() else path
        dest = args.output / "executed_source" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())

    def save():
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    torch.manual_seed(17)
    for index, (shape, lengths) in enumerate(SHAPES.items()):
        maximum = max(lengths)
        operands = [
            torch.randn(
                sum(lengths),
                h,
                256,
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            for h in (16, 4, 4)
        ]
        grad = torch.randn_like(operands[0])
        cuts = torch.tensor((0, *accumulate(lengths)), device="cuda", dtype=torch.int32)
        functions = [
            ("fa4", fa4),
            ("old_mxfp8", old_attention),
            ("fused_dual", fused_attention),
            ("fused_square", functools.partial(fused_attention, square=True)),
        ]
        if index % 2:
            functions.reverse()
        for backend, function in functions:

            def action(
                fn=function,
                operands=operands,
                cuts=cuts,
                maximum=maximum,
                grad=grad,
            ):
                out = fn(*operands, cuts, maximum)
                derivatives = torch.autograd.grad(out, operands, grad)
                return out, derivatives

            print(f"warming {shape} {backend}", flush=True)
            for _ in range(6):
                out, derivatives = action()
            if not bool(
                torch.isfinite(out).all()
                and all(torch.isfinite(x).all() for x in derivatives)
            ):
                raise RuntimeError("nonfinite benchmark output/gradient")
            row = {
                "shape": shape,
                "lengths": lengths,
                "backend": backend,
                "uncaptured": timing(action),
                "graph_replay": graph_timing(
                    function, operands, cuts, max(lengths), grad
                ),
            }
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as profiler:
                action()
                torch.cuda.synchronize()
            trace = args.output / f"{shape}_{backend}_trace.json"
            profiler.export_chrome_trace(str(trace))
            row.update(attribute_kernels(json.loads(trace.read_text())["traceEvents"]))
            row["trace_sha256"] = hashlib.sha256(trace.read_bytes()).hexdigest()
            report["rows"].append(row)
            save()
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in row.items()
                        if k not in {"kernels", "gpu_kernel_stages"}
                    }
                ),
                flush=True,
            )
    report["status"] = "complete"
    save()


if __name__ == "__main__":
    main()
