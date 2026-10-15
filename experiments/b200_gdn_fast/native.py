"""Zero-state pool path versus gathered zero state: bitwise, CP and non-CP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from flashinfer import gdn_prefill

    torch.manual_seed(5)
    report = {"passed": False, "gpu": torch.cuda.get_device_name(), "cases": []}
    heads, dim, pool = 32, 128, 64
    for tokens, seqs, use_cp in (
        (1024, 1, False),
        (32768, 1, "auto"),
        (6000, 3, False),
        (10000, 2, True),
        (16384, 4, True),
        (9000, 3, "auto"),
    ):
        lengths = torch.tensor([tokens // seqs] * seqs)
        lengths[-1] += tokens - lengths.sum()
        cu = torch.cat([torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)]).cuda()
        q = torch.nn.functional.normalize(
            torch.randn(tokens, 16, dim, device="cuda"), dim=-1
        ).bfloat16()
        k = torch.nn.functional.normalize(
            torch.randn(tokens, 16, dim, device="cuda"), dim=-1
        ).bfloat16()
        v = torch.randn(tokens, heads, dim, device="cuda").bfloat16()
        g = torch.exp(-torch.rand(tokens, heads, device="cuda"))
        beta = torch.rand(tokens, heads, device="cuda")
        indices = torch.randperm(pool, device="cuda")[:seqs].to(torch.int32)
        common = dict(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            cu_seqlens=cu,
            output_final_state=True,
            backend="flashinfer",
            use_cp=use_cp,
        )
        reference_out = torch.empty_like(v)
        zero = torch.zeros(seqs, heads, dim, dim, device="cuda")
        _, reference_state = gdn_prefill.chunk_gated_delta_rule(
            **common, initial_state=zero, output=reference_out
        )
        pool_state = torch.randn(pool, heads, dim, dim, device="cuda")
        untouched = pool_state.clone()
        out = torch.empty_like(v)
        gdn_prefill.chunk_gated_delta_rule(
            **common,
            initial_state=None,
            output=out,
            output_state=pool_state,
            state_indices=indices,
        )
        torch.cuda.synchronize()
        mask = torch.ones(pool, dtype=torch.bool, device="cuda")
        mask[indices.long()] = False
        case = {
            "tokens": tokens,
            "seqs": seqs,
            "use_cp": use_cp,
            "output_bitwise": bool(torch.equal(out, reference_out)),
            "state_bitwise": bool(
                torch.equal(pool_state[indices.long()], reference_state)
            ),
            "other_slots_untouched": bool(
                torch.equal(pool_state[mask], untouched[mask])
            ),
        }
        report["cases"].append(case)
        print("gdn_fast_case", case, flush=True)
    report["passed"] = all(
        c["output_bitwise"] and c["state_bitwise"] and c["other_slots_untouched"]
        for c in report["cases"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
