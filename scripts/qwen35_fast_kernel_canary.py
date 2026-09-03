#!/usr/bin/env python3
"""Exercise the Qwen3.5 FLA and causal-conv1d training fast paths."""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import time
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="compare causal-conv1d with the exact Transformers torch fallback",
    )
    parser.add_argument("--warmup-iterations", type=int, default=3)
    parser.add_argument("--benchmark-iterations", type=int, default=10)
    return parser.parse_args()


def benchmark_call(
    torch: Any,
    call: Any,
    *,
    device: Any,
    warmup_iterations: int,
    benchmark_iterations: int,
) -> dict[str, float]:
    """Return synchronized forward/backward timings for one callable."""
    for _ in range(warmup_iterations):
        call()
    torch.cuda.synchronize(device)
    timings_ms = []
    for _ in range(benchmark_iterations):
        started = time.perf_counter()
        call()
        torch.cuda.synchronize(device)
        timings_ms.append((time.perf_counter() - started) * 1000)
    return {
        "median_ms": round(statistics.median(timings_ms), 3),
        "mean_ms": round(statistics.mean(timings_ms), 3),
        "min_ms": round(min(timings_ms), 3),
    }


def main() -> None:
    args = parse_args()

    # Import only after the caller has activated the isolated fast-kernel paths.
    import causal_conv1d
    import fla
    import torch
    from transformers.models.qwen3_5.configuration_qwen3_5 import (
        Qwen3_5TextConfig,
    )
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5GatedDeltaNet,
    )
    from transformers.utils.import_utils import (
        is_causal_conv1d_available,
        is_flash_linear_attention_available,
    )

    config = Qwen3_5TextConfig(
        hidden_size=2560,
        intermediate_size=9216,
        num_hidden_layers=32,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=256,
        linear_num_value_heads=32,
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention"] * 32,
    )
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(0)

    layer = Qwen3_5GatedDeltaNet(config, layer_idx=0).to(
        device=device,
        dtype=torch.bfloat16,
    )
    if layer.causal_conv1d_fn is None:
        raise RuntimeError("Qwen3.5 did not bind the causal-conv1d fast path")

    causal_calls = 0
    causal_conv = layer.causal_conv1d_fn

    def counted_causal_conv(*call_args: Any, **call_kwargs: Any) -> torch.Tensor:
        nonlocal causal_calls
        causal_calls += 1
        return causal_conv(*call_args, **call_kwargs)

    layer.causal_conv1d_fn = counted_causal_conv
    hidden_states = torch.randn(
        1,
        args.sequence_length,
        config.hidden_size,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    output = layer(hidden_states=hidden_states)
    output.float().square().mean().backward()
    torch.cuda.synchronize(device)
    if causal_calls != 1:
        raise RuntimeError(f"expected one causal-conv1d call, observed {causal_calls}")

    result = {
        "status": "PASS",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "fla": fla.__version__,
        "causal_conv1d": causal_conv1d.__version__,
        "transformers_fla_available": is_flash_linear_attention_available(),
        "transformers_causal_conv1d_available": is_causal_conv1d_available(),
        "causal_conv1d_calls": causal_calls,
        "sequence_length": args.sequence_length,
        "nvcc": shutil.which("nvcc"),
        "peak_allocated_gib": round(
            torch.cuda.max_memory_allocated(device) / 1024**3,
            3,
        ),
    }
    if args.benchmark:
        import torch.nn.functional as functional

        conv_input = torch.randn(
            1,
            layer.conv_dim,
            args.sequence_length,
            device=device,
            dtype=torch.bfloat16,
        )
        conv_weight = torch.randn(
            layer.conv_dim,
            config.linear_conv_kernel_dim,
            device=device,
            dtype=torch.bfloat16,
        )

        def causal_step() -> None:
            x = conv_input.detach().requires_grad_(True)
            weight = conv_weight.detach().requires_grad_(True)
            value = causal_conv(
                x=x,
                weight=weight,
                bias=None,
                activation=config.hidden_act,
            )
            torch.autograd.grad(value.float().square().mean(), (x, weight))

        def torch_step() -> None:
            x = conv_input.detach().requires_grad_(True)
            weight = conv_weight.detach().requires_grad_(True)
            value = functional.silu(
                functional.conv1d(
                    x,
                    weight.unsqueeze(1),
                    groups=layer.conv_dim,
                    padding=config.linear_conv_kernel_dim - 1,
                )[..., : args.sequence_length]
            )
            torch.autograd.grad(value.float().square().mean(), (x, weight))

        def layer_step(*, use_causal_conv: bool) -> None:
            layer.zero_grad(set_to_none=True)
            layer.causal_conv1d_fn = causal_conv if use_causal_conv else None
            value = layer(hidden_states=hidden_states.detach().requires_grad_(True))
            value.float().square().mean().backward()

        causal_timing = benchmark_call(
            torch,
            causal_step,
            device=device,
            warmup_iterations=args.warmup_iterations,
            benchmark_iterations=args.benchmark_iterations,
        )
        torch_timing = benchmark_call(
            torch,
            torch_step,
            device=device,
            warmup_iterations=args.warmup_iterations,
            benchmark_iterations=args.benchmark_iterations,
        )
        causal_layer_timing = benchmark_call(
            torch,
            lambda: layer_step(use_causal_conv=True),
            device=device,
            warmup_iterations=args.warmup_iterations,
            benchmark_iterations=args.benchmark_iterations,
        )
        torch_layer_timing = benchmark_call(
            torch,
            lambda: layer_step(use_causal_conv=False),
            device=device,
            warmup_iterations=args.warmup_iterations,
            benchmark_iterations=args.benchmark_iterations,
        )
        result["conv_training_benchmark"] = {
            "shape": list(conv_input.shape),
            "causal_conv1d": causal_timing,
            "torch_fallback": torch_timing,
            "speedup_median": round(
                torch_timing["median_ms"] / causal_timing["median_ms"],
                3,
            ),
        }
        result["gated_delta_layer_training_benchmark"] = {
            "causal_conv1d": causal_layer_timing,
            "torch_fallback": torch_layer_timing,
            "speedup_median": round(
                torch_layer_timing["median_ms"]
                / causal_layer_timing["median_ms"],
                3,
            ),
        }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
