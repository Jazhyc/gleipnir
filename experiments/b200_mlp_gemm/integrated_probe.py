"""Complete FP4 LoRA MLP forward/backward, ordinary dispatch and graph replay."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import traceback
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
from transformers import AutoConfig

import gleipnir.cudnn_fp4_mlp as native
from experiments.b200_mlp_gemm.probe import make_mlp, relative_l2
from gleipnir.bf16_lora import configure_bf16_reductions
from gleipnir.cudnn_fp4_gemm import decode_operand, pack_operand


def decoded_linear(x, weight, other, backward):
    with torch.no_grad():
        pair = native.prepare_weights(weight, other)
        b = pair.backward if backward else pair.forward
        flat = x.reshape(-1, x.shape[-1]).contiguous()
        a = pack_operand(flat, row_amax=True, chunked_rows=True)
        av = decode_operand(a) * a.inverse.reshape(-1, 1)
        bv = decode_operand(b) * b.inverse.reshape(-1, 1)
        raw = F.linear(av, bv).to(torch.bfloat16)
        result = (raw.float() * (1 / (a.inverse[:, None] * b.inverse))).to(
            torch.bfloat16
        )
        return result.reshape(*x.shape[:-1], b.codes.shape[0])


def timings(functions):
    samples = {k: [] for k in functions}
    for repeat in range(10):
        order = list(functions)
        if repeat % 2:
            order.reverse()
        for name in order:
            torch.cuda.synchronize()
            start = time.perf_counter()
            functions[name]()
            torch.cuda.synchronize()
            samples[name].append((time.perf_counter() - start) * 1000)
    return {
        "samples_ms": samples,
        "mean_ms": {k: statistics.mean(v) for k, v in samples.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = args.output / "probe.json"
    report = {
        "status": "starting",
        "variant": "fp4_integrated",
        "shapes": [],
        "full_backward_included": True,
        "synthetic_weights": True,
        "timing_baseline": "compiled_peft",
        "full_training": False,
    }

    def save():
        receipt.write_text(json.dumps(report, indent=2) + "\n")

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
        m = make_mlp(config)
        original = m.forward
        parameters = [p for p in m.parameters() if p.requires_grad]
        report["installation"] = native.install_fp4_mlp(m)
        candidate = m.forward
        compiled = {
            "baseline": torch.compile(original, fullgraph=True, dynamic=True),
            "candidate": torch.compile(candidate, fullgraph=True, dynamic=True),
        }
        report["runtime"] = {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
        }
        for tokens in (193, 4096, 16384):
            x = torch.randn(
                1,
                tokens,
                config.hidden_size,
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            dy = torch.randn_like(x) / math.sqrt(config.hidden_size)

            def step(fn, x=x, dy=dy):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    y = fn(x)
                return y, torch.autograd.grad(y, (x, *parameters), dy)

            y, grads = step(candidate)
            with patch.object(native, "_native_linear", decoded_linear):
                ref, ref_grads = step(candidate)
            bf16_y, bf16_grads = step(original)
            changed = x.detach().clone()
            changed[:, 17:] *= 31.7
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                isolated = candidate(changed)
            row = {
                "tokens": tokens,
                "oracle_output_relative_l2": relative_l2(y, ref),
                "oracle_gradient_relative_l2": [
                    relative_l2(a, b) for a, b in zip(grads, ref_grads, strict=True)
                ],
                "bf16_output_relative_l2": relative_l2(y, bf16_y),
                "bf16_gradient_relative_l2": [
                    relative_l2(a, b) for a, b in zip(grads, bf16_grads, strict=True)
                ],
                "finite": all(bool(torch.isfinite(t).all()) for t in (y, *grads)),
                "row_isolation_relative_l2": relative_l2(isolated[:, :17], y[:, :17]),
            }
            report["shapes"].append(row)
            save()
            print(json.dumps(row), flush=True)
            if (
                not row["finite"]
                or row["oracle_output_relative_l2"] > 0.01
                or max(row["oracle_gradient_relative_l2"]) > 0.02
                or row["row_isolation_relative_l2"] != 0
            ):
                raise ValueError("integrated FP4 MLP oracle/isolation gate failed")
            del y, grads, ref, ref_grads, bf16_y, bf16_grads, changed, isolated
            for _ in range(6):
                for fn in compiled.values():
                    step(fn)
            row["ordinary_timing"] = timings(
                {k: lambda fn=fn: step(fn) for k, fn in compiled.items()}
            )
            graphs = {}
            outputs = {}
            warm = torch.cuda.Stream()
            warm.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warm):
                for fn in compiled.values():
                    for _ in range(3):
                        step(fn)
            torch.cuda.current_stream().wait_stream(warm)
            torch.cuda.synchronize()
            for name, fn in compiled.items():
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    outputs[name] = step(fn)
                graphs[name] = g
            source = x.detach().clone()

            def replay(g, x=x, source=source):
                with torch.no_grad():
                    x.copy_(source)
                g.replay()

            for _ in range(6):
                for g in graphs.values():
                    replay(g)
            row["graph_timing"] = timings(
                {k: lambda g=g: replay(g) for k, g in graphs.items()}
            )
            row["graph_copy_included"] = True
            # Replay must read changed inputs and the live FP32 adapter master.
            saved_master = parameters[1].detach().clone()
            with torch.no_grad():
                source.mul_(0.7)
                parameters[1].mul_(0.8)
            replay(graphs["candidate"])
            torch.cuda.synchronize()
            live_y, live_grads = step(compiled["candidate"])
            graph_y, graph_grads = outputs["candidate"]
            row["changed_input_master_replay_relative_l2"] = [
                relative_l2(graph_y, live_y),
                *[
                    relative_l2(a, b)
                    for a, b in zip(graph_grads, live_grads, strict=True)
                ],
            ]
            if max(row["changed_input_master_replay_relative_l2"]) > 0.01:
                raise ValueError("live master/input CUDA graph replay failed")
            with torch.no_grad():
                parameters[1].copy_(saved_master)
            means = row["graph_timing"]["mean_ms"]
            row["relative_graph_improvement"] = (
                1 - means["candidate"] / means["baseline"]
            )
            row["cache"] = native.cache_metadata()
            save()
            print(json.dumps(row), flush=True)
            del (
                graphs,
                outputs,
                g,
                graph_y,
                graph_grads,
                live_y,
                live_grads,
                x,
                dy,
                source,
            )
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
