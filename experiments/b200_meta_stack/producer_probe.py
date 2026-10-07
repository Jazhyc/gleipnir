"""Fused norm/RoPE producer forward, backward, isolation and warmed timing."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import traceback
from contextlib import ExitStack
from dataclasses import asdict
from itertools import accumulate
from pathlib import Path

import torch

from experiments.b200_meta_stack.producer_reference import norm_rope
from experiments.b200_nvidia_mxfp8_varlen import kernel_canary as checks
from experiments.b200_nvidia_mxfp8_varlen.profile_attention import SHAPES, timing
from gleipnir.nvidia_mxfp8_fused_attention import packed_attention
from gleipnir.nvidia_mxfp8_fused_quantize import prepare
from gleipnir.nvidia_mxfp8_meta_variants import NativeVariant, native_variant
from gleipnir.nvidia_mxfp8_norm_rope import _backward, norm_rope_attention


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dq-store-bits", type=int, default=16)
    parser.add_argument("--p-scale-log2", type=int, default=0)
    args = parser.parse_args()
    variant = NativeVariant(
        dq_store_bits=args.dq_store_bits, forward_p_scale_log2=args.p_scale_log2
    )
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "starting", "producer_checks": [], "rows": []}

    def save():
        (args.output / "producer.json").write_text(json.dumps(report, indent=2) + "\n")

    report["spec"] = asdict(variant)
    paths = [
        Path("src/gleipnir/__init__.py"),
        Path("src/gleipnir/_compat.py"),
        Path(__file__),
        *Path("src/gleipnir/kernels/mxfp8").glob("nvidia_mxfp8*.py"),
    ]
    for p in paths:
        dest = args.output / "executed_source" / p
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(p.read_bytes())
    save()
    stack = ExitStack()
    try:
        stack.enter_context(native_variant(variant))
        torch.manual_seed(31)
        lengths = (1, 3, 31, 33, 63, 65, 127, 129, 255, 257)
        total = sum(lengths)
        cu = torch.tensor((0, *accumulate(lengths)), device="cuda", dtype=torch.int32)
        weights = [
            torch.randn(256, device="cuda", dtype=torch.bfloat16) * 0.1
            for _ in range(2)
        ]
        angles = torch.randn(total, 32, device="cuda")
        cos = angles.cos().repeat(1, 2).bfloat16()
        sin = angles.sin().repeat(1, 2).bfloat16()
        for heads in (16, 4):
            slab = torch.randn(total, heads, 512, device="cuda", dtype=torch.bfloat16)
            raw = slab[..., :256].detach().requires_grad_(True)
            weight = weights[0 if heads == 16 else 1]
            reference = norm_rope(raw, weight, cos, sin)
            expected = prepare(reference, cu, max(lengths), square=True)
            actual = prepare(
                raw, cu, max(lengths), square=True, norm_weight=weight, cos=cos, sin=sin
            )
            code_match = float(
                (expected[0].view(torch.uint8) == actual[0].view(torch.uint8))
                .float()
                .mean()
            )
            scales_equal = [
                bool(torch.equal(a, b))
                for a, b in zip(actual[2:], expected[2:], strict=True)
            ]
            # Dead forward capacity is not a live scale contract. The payload
            # and all three native backward scale layouts cover the live oracle.
            grad = torch.randn_like(reference)
            expected_grad = torch.autograd.grad(reference, raw, grad)[0]
            actual_grad = _backward(raw, weight, cos, sin, grad, 1e-6)
            relative = checks.error(actual_grad, expected_grad)
            check = {
                "heads": heads,
                "code_agreement": code_match,
                "native_scales_bitexact": scales_equal[:3],
                "backward_relative_l2": relative,
            }
            report["producer_checks"].append(check)
            save()
            if code_match < 0.999 or not all(scales_equal[:3]) or relative > 0.01:
                raise ValueError("fused producer exceeded 0.1% code / 1% gradient gate")

        q, k, v = [
            torch.randn(
                total, h, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
            )
            for h in (16, 4, 4)
        ]

        def baseline(q, k, v, cu, maximum, cos=cos, sin=sin):
            return packed_attention(
                norm_rope(q, weights[0], cos, sin),
                norm_rope(k, weights[1], cos, sin),
                v,
                cu,
                maximum,
                square=True,
            )

        def candidate(q, k, v, cu, maximum, cos=cos, sin=sin):
            return norm_rope_attention(q, k, v, *weights, cos, sin, cu, maximum)

        go = torch.randn_like(q)
        old = baseline(q, k, v, cu, max(lengths))
        new = candidate(q, k, v, cu, max(lengths))
        oldg = torch.autograd.grad(old, (q, k, v), go)
        newg = torch.autograd.grad(new, (q, k, v), go)
        report["attention_comparison"] = {
            "forward_relative_l2": checks.error(new, old),
            "gradient_relative_l2": [
                checks.error(a, b) for a, b in zip(newg, oldg, strict=True)
            ],
            "finite": bool(torch.isfinite(new).all())
            and all(bool(torch.isfinite(g).all()) for g in newg),
        }
        save()
        if (
            not report["attention_comparison"]["finite"]
            or max(report["attention_comparison"]["gradient_relative_l2"]) > 0.01
            or report["attention_comparison"]["forward_relative_l2"] > 0.01
        ):
            raise ValueError("producer attention changed beyond 1% integration gate")
        # Perturb only the second example, including its rotary tables.
        pivot = lengths[0]
        changed = [x.detach().clone() for x in (q, k, v)]
        for x in changed:
            x[pivot:] *= 100
            x.requires_grad_(True)
        other = candidate(*changed, cu, max(lengths))
        local_grad = torch.zeros_like(other)
        local_grad[:pivot] = 1
        grads = torch.autograd.grad(other, changed, local_grad)
        leakage = max(float(g[pivot:].abs().max()) for g in grads)
        output_difference = float((other[:pivot] - new[:pivot]).abs().max())
        report["isolation"] = {
            "cross_input_gradient_max_abs": leakage,
            "perturb_max_abs": output_difference,
            "passed": leakage == 0 and output_difference == 0,
        }
        if not report["isolation"]["passed"]:
            raise ValueError("producer isolation failed")
        save()
        # Existing native replay/poison gates remain separate. Graph producer
        # replay checks data changes with fixed tables and dynamic cuts.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            gq, gk, gv = [x.detach().clone().requires_grad_() for x in (q, k, v)]
            go = go.clone()
            for _ in range(2):
                warm = candidate(gq, gk, gv, cu, max(lengths))
                torch.autograd.grad(warm, (gq, gk, gv), go)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            gout = candidate(gq, gk, gv, cu, max(lengths))
            gderivatives = torch.autograd.grad(gout, (gq, gk, gv), go)
        changed_cuts = list(accumulate((0, *lengths)))
        changed_cuts[1] += 1
        cu.copy_(torch.tensor(changed_cuts, device="cuda", dtype=torch.int32))
        graph.replay()
        torch.cuda.synchronize()
        eager = candidate(gq, gk, gv, cu, max(lengths))
        eager_g = torch.autograd.grad(eager, (gq, gk, gv), go)
        errors = [
            checks.error(a, b) for a, b in zip(gderivatives, eager_g, strict=True)
        ]
        report["graph_replay"] = {
            "forward_relative_l2": checks.error(gout, eager),
            "gradient_relative_l2": errors,
            "changed_device_lengths": True,
            "passed": checks.error(gout, eager) == 0 and all(e == 0 for e in errors),
        }
        cu.copy_(
            torch.tensor((0, *accumulate(lengths)), device="cuda", dtype=torch.int32)
        )
        if not report["graph_replay"]["passed"]:
            raise ValueError("producer changed-cut graph replay failed")
        save()

        for shape, lengths in SHAPES.items():
            total = sum(lengths)
            cuts = torch.tensor(
                (0, *accumulate(lengths)), device="cuda", dtype=torch.int32
            )
            operands = [
                torch.randn(
                    total,
                    h,
                    256,
                    device="cuda",
                    dtype=torch.bfloat16,
                    requires_grad=True,
                )
                for h in (16, 4, 4)
            ]
            grad = torch.randn_like(operands[0])
            angles = torch.randn(total, 32, device="cuda")
            cs = angles.cos().repeat(1, 2).bfloat16()
            sn = angles.sin().repeat(1, 2).bfloat16()
            for backend, function in (
                ("separate", baseline),
                ("fused_norm_rope", candidate),
            ):
                fn = functools.partial(function, cos=cs, sin=sn)
                maximum = max(lengths)

                def action(fn=fn, xs=operands, cuts=cuts, maximum=maximum, grad=grad):
                    out = fn(*xs, cuts, maximum)
                    return out, torch.autograd.grad(out, xs, grad)

                for _ in range(6):
                    action()
                row = {"shape": shape, "backend": backend, "timing": timing(action)}
                report["rows"].append(row)
                save()
                print(json.dumps(row), flush=True)
        report["status"] = "complete"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
        raise
    finally:
        stack.close()
        report["source_sha256"] = {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path("src/gleipnir/__init__.py"),
                Path("src/gleipnir/_compat.py"),
                Path(__file__),
                *Path("src/gleipnir/kernels/mxfp8").glob("nvidia_mxfp8_norm_rope*.py"),
                Path("src/gleipnir/kernels/mxfp8/nvidia_mxfp8_fused_quantize.py"),
            ]
        }
        save()


if __name__ == "__main__":
    main()
