"""Isolated timing: vLLM conv1d + post-conv versus the fused kernel."""

from __future__ import annotations

import statistics


def bench(name: str, lengths: list[int]) -> None:
    import torch
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        fused_post_conv_prep,
    )
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn

    from gleipnir.serving.gdn_conv_fused import fused_conv_post, token_map

    H, HV, K, V = 16, 32, 128, 128
    dim = 2 * H * K + HV * V
    tokens = sum(lengths)
    mixed = torch.randn(tokens, 12288, device="cuda", dtype=torch.bfloat16)[:, :dim]
    weight = torch.randn(dim, 4, device="cuda", dtype=torch.bfloat16)
    state = torch.randn(len(lengths) + 1, 3, dim, device="cuda", dtype=torch.bfloat16)
    idx = torch.arange(1, len(lengths) + 1, dtype=torch.int32, device="cuda")
    init = torch.zeros(len(lengths), dtype=torch.bool, device="cuda")
    qsl = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        dtype=torch.int32,
        device="cuda",
    )
    a = torch.randn(tokens, HV, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(tokens, HV, device="cuda", dtype=torch.bfloat16)
    A_log = torch.randn(HV, device="cuda")
    dt = torch.randn(HV, device="cuda")
    conv_state = state.transpose(-1, -2)

    def upstream():
        conv = causal_conv1d_fn(
            mixed.transpose(0, 1),
            weight,
            None,
            activation="silu",
            conv_states=conv_state,
            has_initial_state=init,
            cache_indices=idx,
            query_start_loc=qsl,
        ).transpose(0, 1)
        fused_post_conv_prep(
            conv_output=conv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt,
            num_k_heads=H,
            head_k_dim=K,
            head_v_dim=V,
            apply_l2norm=True,
            output_g_exp=False,
        )

    seq_pos = token_map(qsl, tokens)

    def fused():
        fused_conv_post(
            mixed,
            weight,
            None,
            conv_state,
            idx,
            init,
            qsl,
            a,
            b,
            A_log,
            dt,
            H,
            K,
            V,
            seq_pos=seq_pos,
        )

    def timed(fn):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        values = []
        for _ in range(20):
            start.record()
            fn()
            end.record()
            end.synchronize()
            values.append(start.elapsed_time(end))
        return statistics.median(values)

    up, fu = timed(upstream), timed(fused)
    print(
        f"conv_bench {name}: up {up:.3f} ms fused {fu:.3f} ms x{fu / up:.3f}",
        flush=True,
    )


def main() -> None:
    for name, lengths in (
        ("1k", [1024]),
        ("8k", [8192]),
        ("32k", [32768]),
        ("128x40", [40] * 128),
    ):
        bench(name, lengths)


if __name__ == "__main__":
    main()
