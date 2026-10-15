"""Strided gated RMS norm versus upstream's flattened copy: bitwise on GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from vllm.third_party.flash_linear_attention.ops.layernorm_guard import (
        layer_norm_fwd,
    )

    from gleipnir.serving import gdn_fast

    torch.manual_seed(11)
    heads, dim, eps = 32, 128, 1e-6
    report = {"gpu": torch.cuda.get_device_name(), "cases": []}
    for norm_before_gate, activation in (
        (False, "silu"),
        (True, "swish"),
        (False, "swish"),
    ):
        weight = torch.randn(dim, device="cuda", dtype=torch.bfloat16)
        layer = SimpleNamespace(
            norm=SimpleNamespace(
                weight=weight,
                bias=None,
                eps=eps,
                group_size=None,
                norm_before_gate=norm_before_gate,
                activation=activation,
            )
        )
        for tokens in (1, 7, 17, 200, 889, 1024, 4096, 32768):
            packed = torch.randn(tokens, 12288, device="cuda", dtype=torch.bfloat16)
            gate = packed[:, 8192:]
            x = torch.randn(tokens, heads, dim, device="cuda", dtype=torch.bfloat16)
            reference = x.clone()
            layer_norm_fwd(
                reference.reshape(-1, dim),
                weight.contiguous(),
                None,
                eps,
                z=gate.reshape(tokens, heads, dim).reshape(-1, dim),
                out=reference.reshape(-1, dim),
                group_size=dim,
                norm_before_gate=norm_before_gate,
                is_rms_norm=True,
                activation=activation,
            )
            candidate = x.clone()
            used = gdn_fast.strided_gated_norm(layer, candidate, gate, candidate)
            torch.cuda.synchronize()
            case = {
                "tokens": tokens,
                "norm_before_gate": norm_before_gate,
                "activation": activation,
                "strided_path": used,
                "bitwise": bool(torch.equal(reference, candidate)),
                "max_abs": float((reference.float() - candidate.float()).abs().max()),
            }
            report["cases"].append(case)
            print("norm_case", case, flush=True)
            gdn_fast._NORM_WEIGHTS.clear()
    report["passed"] = all(c["bitwise"] and c["strided_path"] for c in report["cases"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("norm_probe_passed", report["passed"], flush=True)


if __name__ == "__main__":
    main()
