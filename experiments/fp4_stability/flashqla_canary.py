"""Compare Qwen4B-shaped BF16 Gated DeltaNet outputs, gradients and warm times."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as functional
from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_kernel

from gleipnir.flashqla_training import (
    BOUNDARY_POLICIES,
    load_flashqla,
    make_flashqla_kernel,
    make_precision_boundary,
    tensor_comparison,
)
from gleipnir.monitoring_systems_screen import sha256_file

SHAPES = [(1, 257), (1, 4096), (1, 16384), (1, 29696), (2, 8192), (4, 4096), (8, 2048)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--auto-cp", action="store_true")
    parser.add_argument("--bf16-boundary", action="store_true")
    parser.add_argument(
        "--boundary-policy", choices=sorted(BOUNDARY_POLICIES), default="bf16"
    )
    args = parser.parse_args()
    if args.boundary_policy != "bf16" and not args.bf16_boundary:
        parser.error("boundary policy requires --bf16-boundary")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    function, receipt = load_flashqla()
    candidate = make_flashqla_kernel(function, auto_cp=args.auto_cp)
    if args.bf16_boundary:
        candidate = make_precision_boundary(candidate, policy=args.boundary_policy)
    fla_bf16 = (
        make_precision_boundary(fla_kernel, policy=args.boundary_policy)
        if args.bf16_boundary
        else None
    )
    report = dict(
        status="running",
        backend=receipt,
        auto_cp=args.auto_cp,
        bf16_boundary=args.bf16_boundary,
        boundary_policy=args.boundary_policy,
        input_dtype="float32" if args.bf16_boundary else "bfloat16",
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        source_sha256=sha256_file(Path(__file__)),
        helper_sha256=sha256_file(Path("src/gleipnir/training/backends/flashqla.py")),
        output_relative_l2_gate=0.01,
        gradient_relative_l2_gate=0.02,
        warmup_calls=10,
        measured_calls=30,
        scope="isolated forward and forward+backward; no model or optimizer",
        cases=[],
    )

    def publish():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    publish()
    try:
        for case_index, (batch, length) in enumerate(SHAPES):
            torch.manual_seed(10 + case_index)
            shape = (batch, length, 32, 128)
            dtype = torch.float32 if args.bf16_boundary else torch.bfloat16
            q, k, v = [torch.randn(shape, device="cuda", dtype=dtype) for _ in range(3)]
            a = torch.randn(shape[:-1], device="cuda", dtype=dtype)
            g = -functional.softplus(a.float())
            beta = torch.randn_like(a).sigmoid()
            # Include zeroed tails, as masked hidden states feed the padded recipe.
            if batch > 1:
                q[-1, length * 3 // 4 :] = 0
                k[-1, length * 3 // 4 :] = 0
                v[-1, length * 3 // 4 :] = 0
            inputs = [x.requires_grad_() for x in [q, k, v, g, beta]]
            do = torch.randn_like(v)

            def run(kernel, *, backward: bool, inputs=inputs, do=do):
                o, state = kernel(
                    *inputs,
                    initial_state=None,
                    output_final_state=False,
                    use_qk_l2norm_in_kernel=True,
                )
                if state is not None:
                    raise ValueError("unexpected final state")
                gradients = torch.autograd.grad(o, inputs, do) if backward else None
                return o, gradients

            case = dict(batch=batch, length=length, heads=32, head_dim=128)
            report["cases"].append(case)
            reference, reference_gradients = run(fla_kernel, backward=True)
            actual, gradients = run(candidate, backward=True)
            case["output"] = tensor_comparison(actual, reference)
            case["gradients"] = {
                name: tensor_comparison(a, r)
                for name, a, r in zip(
                    ["q", "k", "v", "g", "beta"],
                    gradients,
                    reference_gradients,
                    strict=True,
                )
            }
            passed = case["output"]["finite"] and case["output"]["relative_l2"] <= 0.01
            passed = passed and all(
                x["finite"] and x["relative_l2"] <= 0.02
                for x in case["gradients"].values()
            )
            case["passed"] = passed
            if fla_bf16 is not None:
                cast_output, cast_gradients = run(fla_bf16, backward=True)
                cast_metrics = [tensor_comparison(cast_output, reference)] + [
                    tensor_comparison(a, r)
                    for a, r in zip(cast_gradients, reference_gradients, strict=True)
                ]
                case["fla_bf16_comparison"] = cast_metrics
                case["passed"] = passed = passed and all(
                    x["finite"] and x["relative_l2"] <= (0.01 if i == 0 else 0.02)
                    for i, x in enumerate(cast_metrics)
                )
                del cast_output, cast_gradients
            publish()
            print(
                f"case={batch}x{length} parity={passed} "
                f"output_l2={case['output']['relative_l2']}",
                flush=True,
            )
            if not passed:
                raise ValueError("FlashQLA isolated output/gradient gate failed")
            del reference, actual, reference_gradients, gradients
            case["timing"] = {}
            for backward in [False, True]:
                phase = "forward_backward" if backward else "forward"
                case["timing"][phase] = {}
                # Alternate order across shapes to limit systematic ordering effects.
                pairs = [("fla", fla_kernel), ("flashqla", candidate)]
                if fla_bf16 is not None:
                    pairs.append(("fla_bf16", fla_bf16))
                if case_index % 2:
                    pairs.reverse()
                for name, kernel in pairs:
                    for _ in range(10):
                        run(kernel, backward=backward)
                    torch.cuda.synchronize()
                    durations = []
                    for _ in range(30):
                        started = time.perf_counter()
                        run(kernel, backward=backward)
                        torch.cuda.synchronize()
                        durations.append(time.perf_counter() - started)
                    case["timing"][phase][name] = dict(
                        seconds=durations,
                        mean_seconds=statistics.mean(durations),
                        median_seconds=statistics.median(durations),
                    )
                means = {
                    name: value["mean_seconds"]
                    for name, value in case["timing"][phase].items()
                }
                print(f"case={batch}x{length} {phase}={means}", flush=True)
                publish()
            del run, inputs, q, k, v, g, beta, a, do
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        publish()


if __name__ == "__main__":
    main()
