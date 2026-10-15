"""Fused GDN conv + post-conv versus vLLM's two kernels: bitwise on GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        fused_post_conv_prep,
    )
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn

    from gleipnir.serving.gdn_conv_fused import fused_conv_post

    torch.manual_seed(23)
    H, HV, K, V = 16, 32, 128, 128
    dim = 2 * H * K + HV * V
    report = {"gpu": torch.cuda.get_device_name(), "cases": []}
    cases = [
        ("single_1k_fresh", [1024], [False]),
        ("single_32k_fresh", [32768], [False]),
        ("single_32k_continued", [32768], [True]),
        (
            "mixed_varlen",
            [700, 1, 2, 3, 4096, 17],
            [False, True, True, False, True, False],
        ),
        ("short_with_state", [1, 2, 5], [True, True, True]),
        (
            "many_short",
            [5 + i % 40 for i in range(128)],
            [i % 3 == 0 for i in range(128)],
        ),
        ("null_slot", [9, 12, 7], [True, True, False]),
    ]
    for name, lengths, init in cases:
        tokens = sum(lengths)
        packed = torch.randn(tokens, 12288, device="cuda", dtype=torch.bfloat16)
        mixed = packed[:, :dim]
        weight = torch.randn(dim, 4, device="cuda", dtype=torch.bfloat16) * 0.5
        cache_lines = len(lengths) + 7
        # conv_state [cache, dim, state_len] as a transposed DS-layout view.
        state_storage = torch.randn(
            cache_lines, 3, dim, device="cuda", dtype=torch.bfloat16
        )
        # vLLM reserves cache block 0 as the null block; real requests never use it.
        indices = (1 + torch.randperm(cache_lines - 1, device="cuda"))[
            : len(lengths)
        ].to(torch.int32)
        if name == "null_slot":
            indices[1] = 0
        has_init = torch.tensor(init, device="cuda")
        qsl = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()],
            dtype=torch.int32,
            device="cuda",
        )
        a = torch.randn(tokens, HV, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(tokens, HV, device="cuda", dtype=torch.bfloat16)
        A_log = torch.randn(HV, device="cuda", dtype=torch.float32)
        dt_bias = torch.randn(HV, device="cuda", dtype=torch.float32)

        ref_storage = state_storage.clone()
        ref_state = ref_storage.transpose(-1, -2)
        conv = causal_conv1d_fn(
            mixed.transpose(0, 1),
            weight,
            None,
            activation="silu",
            conv_states=ref_state,
            has_initial_state=has_init,
            cache_indices=indices,
            query_start_loc=qsl,
        ).transpose(0, 1)
        reference = fused_post_conv_prep(
            conv_output=conv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            num_k_heads=H,
            head_k_dim=K,
            head_v_dim=V,
            apply_l2norm=True,
            output_g_exp=False,
        )
        fused_storage = state_storage.clone()
        fused = fused_conv_post(
            mixed,
            weight,
            None,
            fused_storage.transpose(-1, -2),
            indices,
            has_init,
            qsl,
            a,
            b,
            A_log,
            dt_bias,
            H,
            K,
            V,
        )
        torch.cuda.synchronize()
        names = ("q", "k", "v", "g", "beta")
        equal = {
            n: bool(torch.equal(r, f))
            for n, r, f in zip(names, reference, fused, strict=True)
        }
        equal["conv_state"] = bool(torch.equal(ref_storage, fused_storage))
        if name == "null_slot":
            # Upstream leaves null-block outputs uninitialized; compare the others.
            keep = torch.ones(tokens, dtype=torch.bool, device="cuda")
            keep[lengths[0] : lengths[0] + lengths[1]] = False
            equal = {
                n: bool(torch.equal(r[keep], f[keep]))
                for n, r, f in zip(names, reference, fused, strict=True)
            } | {"conv_state": equal["conv_state"]}
        case = {"name": name, "tokens": tokens, "seqs": len(lengths), **equal}
        case["bitwise"] = all(equal.values())
        report["cases"].append(case)
        print("conv_case", case, flush=True)
    report["passed"] = all(c["bitwise"] for c in report["cases"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("conv_probe_passed", report["passed"], flush=True)


if __name__ == "__main__":
    main()
