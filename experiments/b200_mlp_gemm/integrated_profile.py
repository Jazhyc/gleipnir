"""Kernel-time attribution for the completed whole-MLP integration pilot."""

from __future__ import annotations

import argparse
import json
import traceback
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoConfig

from experiments.b200_mlp_gemm.probe import make_mlp
from gleipnir.bf16_lora import configure_bf16_reductions
from gleipnir.cudnn_fp4_mlp import cache_metadata, install_fp4_mlp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "starting",
        "tokens": 16384,
        "replays_per_leg": 10,
        "full_backward_included": True,
        "graph_copy_included": True,
        "synthetic_weights": True,
        "profiled_kernel_times_are_not_wall_time": True,
        "legs": {},
    }

    def save():
        (args.output / "profile.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        configure_bf16_reductions(allow_reduced_precision=False, allow_split_k=False)
        torch.backends.cuda.matmul.allow_tf32 = False
        from torch._inductor import config as ic

        ic.emulate_precision_casts = True
        torch.manual_seed(41)
        config = AutoConfig.from_pretrained(
            "Qwen/Qwen3.5-4B",
            revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            local_files_only=True,
        ).text_config
        model = make_mlp(config)
        original = model.forward
        install_fp4_mlp(model)
        parameters = [p for p in model.parameters() if p.requires_grad]
        x = torch.randn(
            1,
            report["tokens"],
            config.hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        dy = torch.randn_like(x) / config.hidden_size**0.5
        source = x.detach().clone()
        compiled = {
            "baseline": torch.compile(original, fullgraph=True, dynamic=True),
            "candidate": torch.compile(model.forward, fullgraph=True, dynamic=True),
        }

        def step(fn):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = fn(x)
            return y, torch.autograd.grad(y, (x, *parameters), dy)

        for fn in compiled.values():
            for _ in range(6):
                step(fn)
        warm = torch.cuda.Stream()
        warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm):
            for fn in compiled.values():
                for _ in range(3):
                    step(fn)
        torch.cuda.current_stream().wait_stream(warm)
        torch.cuda.synchronize()
        # Retain outputs through profiling so captured allocation addresses stay live.
        outputs = {}
        graphs = {}
        for name, fn in compiled.items():
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs[name] = step(fn)
            graphs[name] = graph

        def replay(graph):
            with torch.no_grad():
                x.copy_(source)
            graph.replay()

        print("whole-MLP graphs warm; collecting CUDA kernel events", flush=True)
        for name, graph in graphs.items():
            for _ in range(6):
                replay(graph)
            torch.cuda.synchronize()
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as prof:
                for _ in range(report["replays_per_leg"]):
                    replay(graph)
                torch.cuda.synchronize()
            prof.export_chrome_trace(str(args.output / f"{name}_trace.json"))
            kernels = defaultdict(lambda: {"calls": 0, "total_us": 0.0})
            for event in prof.events():
                if event.device_type == torch.autograd.DeviceType.CUDA:
                    row = kernels[event.name]
                    row["calls"] += 1
                    row["total_us"] += event.device_time_total
            if not kernels:
                raise RuntimeError("profiler returned no CUDA events")
            report["legs"][name] = sorted(
                ({"name": k, **v} for k, v in kernels.items()),
                key=lambda row: row["total_us"],
                reverse=True,
            )
            save()
            print(f"profiled {name}: {len(kernels)} unique CUDA events", flush=True)
        report["cache"] = cache_metadata()
        report["status"] = "complete"
    except Exception as exc:
        report.update(
            status="failed", error=repr(exc), traceback=traceback.format_exc()
        )
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
