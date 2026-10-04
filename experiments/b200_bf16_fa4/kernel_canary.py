"""Check the actual BF16 GQA varlen shape before model loading."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
from pathlib import Path

import torch
import torch.nn.functional as functional


def main() -> None:
    from flash_attn.cute import flash_attn_varlen_func

    from gleipnir.flashqla_training import load_flashqla

    # Fail before a large model load if an overlay changes pinned recurrence.
    _, recurrence = load_flashqla()

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(0)
    lengths = [1, 3, 63, 65, 127, 129]
    total = sum(lengths)
    cuts = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    q, k, v = [
        torch.randn(
            total, heads, 256, device="cuda", dtype=torch.bfloat16
        ).requires_grad_()
        for heads in [16, 4, 4]
    ]
    rq, rk, rv = [x.detach().float().requires_grad_() for x in [q, k, v]]
    reference = []
    start = 0
    for length in lengths:
        end = start + length
        reference.append(
            functional.scaled_dot_product_attention(
                rq[start:end].transpose(0, 1),
                rk[start:end].transpose(0, 1),
                rv[start:end].transpose(0, 1),
                is_causal=True,
                enable_gqa=True,
            ).transpose(0, 1)
        )
        start = end
    reference = torch.cat(reference)
    candidate = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cuts,
        cu_seqlens_k=cuts,
        max_seqlen_q=max(lengths),
        max_seqlen_k=max(lengths),
        causal=True,
    )
    if isinstance(candidate, tuple):
        candidate = candidate[0]
    gradient = torch.randn_like(candidate)
    reference.backward(gradient.float())
    candidate.backward(gradient)
    errors = {}
    for name, actual, expected, limit in [
        ("forward", candidate, reference, 0.02),
        ("dq", q.grad, rq.grad, 0.05),
        ("dk", k.grad, rk.grad, 0.05),
        ("dv", v.grad, rv.grad, 0.05),
    ]:
        assert actual is not None and torch.isfinite(actual).all()
        error = float((actual.float() - expected).norm() / expected.norm())
        errors[name] = {"relative_l2": error, "limit": limit}
        assert error <= limit, errors
    report = {
        "passed": True,
        "lengths": lengths,
        "heads": [16, 4],
        "head_dim": 256,
        "errors": errors,
        "kernel_module": flash_attn_varlen_func.__module__,
        "packages": {
            name: metadata.version(name)
            for name in [
                "torch",
                "transformers",
                "flash-attn-4",
                "nvidia-cutlass-dsl",
                "nvidia-cutlass-dsl-libs-cu13",
                "apache-tvm-ffi",
            ]
        },
        "gpu": torch.cuda.get_device_name(),
        "recurrence": recurrence,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
