"""Bounded native varlen execution and byte-layout diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import traceback
from itertools import accumulate
from pathlib import Path
from unittest.mock import patch

import torch

from gleipnir.nvidia_mxfp8_attention import mxfp8_attention as dense_attention
from gleipnir.nvidia_mxfp8_attention import quantize as dense_quantize
from gleipnir.nvidia_mxfp8_varlen_attention import packed_attention
from gleipnir.nvidia_mxfp8_varlen_quantize import quantize


def error(a, b):
    a, b = a.detach().float(), b.detach().float()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def dense_reference(q, k, v, offsets):
    """Separate native dense calls preserve the existing arithmetic oracle."""
    return torch.cat(
        [
            dense_attention(q[a:b], k[a:b], v[a:b], scale=0.0625)
            for a, b in zip(offsets[:-1], offsets[1:], strict=True)
        ]
    )


def check_replay(q, k, v, cuts, maximum, grad, offsets) -> dict:
    """Change device lengths under one retained forward/backward CUDA graph."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        # Fresh leaves avoid AccumulateGrad nodes initialized on the legacy
        # stream by earlier comparisons, which invalidate backward capture.
        q, k, v = [x.detach().clone().requires_grad_() for x in (q, k, v)]
        grad = grad.clone()
        for _ in range(2):
            warm = packed_attention(q, k, v, cuts, maximum)
            torch.autograd.grad(warm, (q, k, v), grad)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = packed_attention(q, k, v, cuts, maximum)
        derivatives = torch.autograd.grad(out, (q, k, v), grad)
    changed = list(offsets)
    changed[1] += 1
    # First singleton grows, next example shrinks; total tokens are unchanged.
    cuts.copy_(torch.tensor(changed, device="cuda", dtype=torch.int32))
    graph.replay()
    torch.cuda.synchronize()
    reference = dense_reference(q, k, v, changed)
    gradients = torch.autograd.grad(reference, (q, k, v), grad)
    errors = [error(a, b) for a, b in zip(derivatives, gradients, strict=True)]
    result = {
        "forward_relative_l2": error(out, reference),
        "gradient_relative_l2": errors,
        "changed_device_lengths": True,
        "passed": error(out, reference) == 0 and all(e == 0 for e in errors),
    }
    cuts.copy_(torch.tensor(offsets, device="cuda", dtype=torch.int32))
    return result


def check_isolation(q, k, v, cuts, maximum, offsets) -> dict:
    """Perturb the next example and differentiate loss on only the first."""
    changed = [x.detach().clone().requires_grad_() for x in (q, k, v)]
    first_start, first_end, second_end = offsets[1:4]
    with torch.no_grad():
        for x in changed:
            x[first_end:second_end].mul_(7).add_(3)
    baseline = packed_attention(q, k, v, cuts, maximum)
    candidate = packed_attention(*changed, cuts, maximum)
    derivatives = torch.autograd.grad(
        candidate[first_start:first_end].float().square().sum(), changed
    )
    leakage = max(
        float(torch.cat((x[:first_start], x[first_end:])).abs().max())
        for x in derivatives
    )
    change = float(
        (candidate[first_start:first_end] - baseline[first_start:first_end]).abs().max()
    )
    return {
        "other_example_output_effect": change,
        "other_example_gradient": leakage,
        "passed": change == 0 and leakage == 0,
    }


def check_fp32(q, k, v, grad, offsets) -> dict:
    """Independent FP32 causal GQA reference retains strict numerical limits."""
    operands = [x.detach().float().requires_grad_() for x in (q, k, v)]
    qr, kr, vr = operands
    parts = []
    for a, b in zip(offsets[:-1], offsets[1:], strict=True):
        scores = (
            torch.einsum("thd,shd->hts", qr[a:b], kr[a:b].repeat_interleave(4, dim=1))
            * 0.0625
        )
        mask = torch.ones(b - a, b - a, device="cuda", dtype=torch.bool).triu(1)
        probability = scores.masked_fill(mask, float("-inf")).softmax(-1)
        parts.append(
            torch.einsum(
                "hts,shd->thd", probability, vr[a:b].repeat_interleave(4, dim=1)
            )
        )
    reference = torch.cat(parts)
    derivatives = torch.autograd.grad(reference, operands, grad.float())
    return reference, derivatives


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    if output.exists():
        raise ValueError("preserve existing canary receipts")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "started",
        "quantizer_checks": [],
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path("src/gleipnir").glob("nvidia_mxfp8_varlen*.py")
        },
        "cache_paths": {k: v for k, v in os.environ.items() if "CACHE" in k},
    }
    archive = output.parent / "executed_source"
    for name in report["source_sha256"]:
        path = Path(name)
        target = archive / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())

    def save():
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)

    save()
    try:
        torch.manual_seed(17)
        lengths = (1, 3, 31, 33, 63, 65, 127, 129, 255, 257)
        offsets = (0, *accumulate(lengths))
        total, maximum, batch = sum(lengths), max(lengths), len(lengths)
        cuts = torch.tensor(offsets, device="cuda", dtype=torch.int32)
        for heads in (16, 4):
            source = (
                torch.randn(total, heads, 256, device="cuda", dtype=torch.bfloat16)
                * 0.5
            )
            for col in (False, True):
                payload, sf, packed = quantize(source, cuts, maximum, col)
                tiles = (maximum + 127) // 128
                capacity = (total + 127) // 128 + batch
                dense_sf = (
                    sf.view(2, batch, heads, tiles, 512)
                    if col
                    else sf.view(batch, heads, tiles, 1024)
                )
                packed_expected = []
                payload_ok, scale_ok = True, True
                for i, (start, end) in enumerate(
                    zip(offsets[:-1], offsets[1:], strict=True)
                ):
                    data_ref, scale_ref = dense_quantize(source[start:end], col)
                    n = (lengths[i] + 127) // 128
                    payload_ok &= torch.equal(
                        payload[start:end].view(torch.uint8), data_ref.view(torch.uint8)
                    )
                    if col:
                        ref = scale_ref.view(2, heads, n, 512)
                        scale_ok &= torch.equal(dense_sf[:, i, :, :n], ref)
                        packed_expected.append(
                            ref.permute(1, 2, 0, 3).reshape(heads, n, 1024)
                        )
                    else:
                        ref = scale_ref.view(heads, n, 1024)
                        scale_ok &= torch.equal(dense_sf[i, :, :n], ref)
                        packed_expected.append(ref)
                expected = torch.cat(packed_expected, dim=1)
                packed_ok = torch.equal(
                    packed.view(heads, capacity, 1024)[:, : expected.shape[1]], expected
                )
                row = {
                    "heads": heads,
                    "columnwise": col,
                    "payload_bitexact": payload_ok,
                    "canonical_scale_bitexact": scale_ok,
                    "packed_scale_bitexact": packed_ok,
                }
                report["quantizer_checks"].append(row)
                save()
                if not all((payload_ok, scale_ok, packed_ok)):
                    raise RuntimeError(
                        "packed quantization differs from pinned producer"
                    )
        report["status"] = "attention_starting"
        save()
        q = torch.randn(
            total, 16, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        k = torch.randn(
            total, 4, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        v = torch.randn_like(k, requires_grad=True)
        grad = torch.randn_like(q)
        out = packed_attention(q, k, v, cuts, maximum)
        out.backward(grad)
        torch.cuda.synchronize()
        dense_q, dense_k, dense_v = (
            x.detach().clone().requires_grad_() for x in (q, k, v)
        )
        control = torch.cat(
            [
                dense_attention(dense_q[a:b], dense_k[a:b], dense_v[a:b], scale=0.0625)
                for a, b in zip(offsets[:-1], offsets[1:], strict=True)
            ]
        )
        control.backward(grad)
        report["native_dense_comparison"] = {
            "forward_relative_l2": error(out, control),
            "gradient_relative_l2": [
                error(a.grad, b.grad)
                for a, b in zip((q, k, v), (dense_q, dense_k, dense_v), strict=True)
            ],
            "finite": bool(
                torch.isfinite(out).all()
                and all(torch.isfinite(x.grad).all() for x in (q, k, v))
            ),
        }
        if (
            not report["native_dense_comparison"]["finite"]
            or any(
                report["native_dense_comparison"][key] != 0
                for key in ["forward_relative_l2"]
            )
            or any(report["native_dense_comparison"]["gradient_relative_l2"])
        ):
            raise RuntimeError(
                "direct varlen arithmetic differs from dense native path"
            )
        report["isolation"] = check_isolation(q, k, v, cuts, maximum, offsets)
        save()
        if not report["isolation"]["passed"]:
            raise RuntimeError("cross-example isolation failed")
        ref, derivatives = check_fp32(q, k, v, grad, offsets)
        forward_error = error(out, ref)
        gradient_errors = [
            error(x.grad, y) for x, y in zip((q, k, v), derivatives, strict=True)
        ]
        report["fp32_comparison"] = {
            "forward_relative_l2": forward_error,
            "gradient_relative_l2": gradient_errors,
            "strict_passed": forward_error <= 0.02 and max(gradient_errors) <= 0.05,
        }
        save()
        report["graph_replay"] = check_replay(q, k, v, cuts, maximum, grad, offsets)
        save()
        if not report["graph_replay"]["passed"]:
            raise RuntimeError("changed-length CUDA graph replay failed")
        original_empty = torch.empty

        def poisoned_empty(*args, **kwargs):
            result = original_empty(*args, **kwargs)
            if result.dtype == torch.uint8:
                result.fill_(255)
            elif result.is_floating_point():
                result.fill_(float("nan"))
            return result

        with patch("torch.empty", poisoned_empty):
            poison = packed_attention(q, k, v, cuts, maximum)
            poison_grads = torch.autograd.grad(poison, (q, k, v), grad)
        report["poisoned_dead_storage"] = {
            "passed": bool(
                torch.isfinite(poison).all()
                and all(torch.isfinite(x).all() for x in poison_grads)
            ),
            "forward_relative_l2": error(poison, out),
            "gradient_relative_l2": [
                error(a, b.grad) for a, b in zip(poison_grads, (q, k, v), strict=True)
            ],
        }
        if not report["poisoned_dead_storage"]["passed"]:
            raise RuntimeError("poisoned dead storage escaped masks")
        report["status"] = "execution_complete"
        save()
    except Exception:
        report["status"] = "failed"
        report["exception"] = traceback.format_exc()
        save()
        raise


if __name__ == "__main__":
    main()
