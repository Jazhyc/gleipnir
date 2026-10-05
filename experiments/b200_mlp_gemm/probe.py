"""Actual PEFT MLP arithmetic and complete forward/backward diagnostic timings."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import statistics
import time
import traceback
from pathlib import Path
from types import MethodType

import torch
import yaml
from peft import LoraConfig
from peft.tuners.lora.layer import Linear
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP

from gleipnir.bf16_lora import configure_bf16_reductions
from gleipnir.cudnn_lora_mlp import cudnn_forward
from gleipnir.mlp_gemm import install_merged_mlp


def relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(
        (actual.float() - expected.float()).norm()
        / expected.float().norm().clamp_min(1e-20)
    )


def make_mlp(config, *, device: str = "cuda") -> Qwen3_5MLP:
    """Build the actual Transformers/PEFT module with nonzero FP32 masters."""
    module = Qwen3_5MLP(config, config.intermediate_size).to(
        device=device, dtype=torch.bfloat16
    )
    module.requires_grad_(False)
    lc = LoraConfig(r=128, lora_alpha=256, lora_dropout=0, bias="none")
    for name in ("gate_proj", "up_proj", "down_proj"):
        base = getattr(module, name)
        layer = Linear(base, "default", config=lc, r=128, lora_alpha=256)
        for p in (layer.lora_A["default"].weight, layer.lora_B["default"].weight):
            p.data = torch.randn(p.shape, device=device, dtype=torch.float32) * 0.01
        setattr(module, name, layer)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=["eager", "compiled", "merged", "merged_compiled", "cudnn"],
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load(Path("experiments/b200_mlp_gemm/config.yaml").read_text())
    report = {
        "status": "starting",
        "variant": args.variant,
        "config": cfg,
        "shapes": [],
        "synthetic_weights": True,
        "full_backward_included": True,
    }
    receipt = args.output / "probe.json"

    def save():
        receipt.write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        torch.manual_seed(41)
        configure_bf16_reductions(allow_reduced_precision=False, allow_split_k=False)
        torch.backends.cuda.matmul.allow_tf32 = False
        from torch._inductor import config as ic

        ic.emulate_precision_casts = True
        mc = AutoConfig.from_pretrained(
            "Qwen/Qwen3.5-4B",
            revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            local_files_only=True,
        ).text_config
        module = make_mlp(mc)
        original = module.forward
        parameters = [p for p in module.parameters() if p.requires_grad]
        report.update(
            hidden_size=mc.hidden_size,
            intermediate_size=mc.intermediate_size,
            gpu=torch.cuda.get_device_name(),
            torch_version=torch.__version__,
            runtime_versions={
                name: importlib.metadata.version(name)
                for name in (
                    "transformers",
                    "peft",
                    "triton",
                    "flash-linear-attention",
                    "fla-core",
                    "nvidia-cudnn-frontend",
                )
            },
            cuda_version=torch.version.cuda,
            compiler_threads=16,
        )
        if args.variant.startswith("merged"):
            report["installation"] = install_merged_mlp(
                module, compile_mlp=args.variant.endswith("compiled")
            )
        elif args.variant == "compiled":
            module.forward = torch.compile(original, fullgraph=True, dynamic=True)
        elif args.variant == "cudnn":
            module.forward = MethodType(cudnn_forward, module)
        candidate = module.forward
        # The production decoder shells already compile their MLP operations.
        timed_reference = (
            torch.compile(original, fullgraph=True, dynamic=True)
            if args.variant in {"merged_compiled", "cudnn"}
            else original
        )
        report["timing_baseline"] = (
            "compiled_peft" if timed_reference is not original else "eager_peft"
        )
        for tokens in cfg["tokens"]:
            x = torch.randn(
                tokens,
                mc.hidden_size,
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            dy = torch.randn(
                tokens, mc.hidden_size, device="cuda", dtype=torch.bfloat16
            ) / math.sqrt(mc.hidden_size)

            def forward(fn, x=x):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    return fn(x)

            def step(fn, x=x, dy=dy):
                y = forward(fn)
                grads = torch.autograd.grad(y, (x, *parameters), dy)
                return y, grads

            print(f"validating {args.variant} tokens={tokens}", flush=True)
            y0, g0 = step(original)
            started = time.perf_counter()
            y1, g1 = step(candidate)
            torch.cuda.synchronize()
            errors = [relative_l2(a, b) for a, b in zip(g1, g0, strict=True)]
            finite = all(bool(torch.isfinite(t).all()) for t in (y1, *g1))
            output_error = relative_l2(y1, y0)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                # Changing other rows cannot affect the first 17 rows.
                changed = x.detach().clone()
                changed[17:] = torch.randn_like(changed[17:])
                isolation = relative_l2(candidate(changed)[:17], y1.detach()[:17])
            row = {
                "tokens": tokens,
                "setup_seconds": time.perf_counter() - started,
                "output_relative_l2": output_error,
                "gradient_relative_l2": errors,
                "finite": finite,
                "row_isolation_relative_l2": isolation,
            }
            report["shapes"].append(row)
            save()
            if (
                not finite
                or max(output_error, *errors) > cfg["relative_l2_limit"]
                or isolation > 1e-7
            ):
                raise ValueError(f"arithmetic/isolation gate failed: {row}")
            del y0, g0, y1, g1, changed
            for _ in range(cfg["warmups"]):
                step(timed_reference)
                step(candidate)
                with torch.no_grad():
                    forward(timed_reference)
                    forward(candidate)
            timings = {"baseline": [], "candidate": []}
            forward_timings = {"baseline": [], "candidate": []}
            for i in range(cfg["repetitions"]):
                order = [("baseline", timed_reference), ("candidate", candidate)]
                if i % 2:
                    order.reverse()
                for name, fn in order:
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    step(fn)
                    torch.cuda.synchronize()
                    timings[name].append((time.perf_counter() - start) * 1000)
                    with torch.no_grad():
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        forward(fn)
                        torch.cuda.synchronize()
                        forward_timings[name].append(
                            (time.perf_counter() - start) * 1000
                        )
            row.update(
                step_ms=timings,
                forward_ms=forward_timings,
                mean_step_ms={k: statistics.mean(v) for k, v in timings.items()},
                mean_forward_ms={
                    k: statistics.mean(v) for k, v in forward_timings.items()
                },
            )
            row["relative_step_improvement"] = (
                1 - row["mean_step_ms"]["candidate"] / row["mean_step_ms"]["baseline"]
            )
            save()
            print(json.dumps(row), flush=True)
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
