"""Check fresh packed attention and zero-start/discard-state GDN equivalence."""

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from types import SimpleNamespace

    import torch
    from flashinfer.gdn_prefill import chunk_gated_delta_rule
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
    from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

    from gleipnir.serving_mxfp8 import paged_forward
    from gleipnir.serving_prompt_only import packed_forward

    torch.manual_seed(73)
    sources = [
        "src/gleipnir/__init__.py",
        "src/gleipnir/_compat.py",
        "src/gleipnir/serving/prompt_only.py",
        "src/gleipnir/serving/prompt_only_contract.py",
        "src/gleipnir/serving/mxfp8.py",
        "experiments/b200_attention_gdn_serving/prompt_only_canary.py",
    ]
    report = {
        "passed": False,
        "gpu": torch.cuda.get_device_name(),
        "checks": [],
        "sources": {
            p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources
        },
    }

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    for lengths in ([1, 3, 127, 129], [17, 65, 513]):
        cu_cpu = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32
        )
        cu = cu_cpu.cuda()
        nums, batch, offsets = compute_causal_conv1d_metadata(cu_cpu, device=cu.device)
        metadata = SimpleNamespace(
            nums_dict=nums, batch_ptr=batch, token_chunk_offset_ptr=offsets
        )
        x = torch.randn(sum(lengths), 64, dtype=torch.bfloat16, device="cuda").T
        w = torch.randn(64, 4, dtype=x.dtype, device=x.device)
        old_state = torch.full(
            (len(lengths) + 1, 64, 3), float("nan"), dtype=x.dtype, device=x.device
        )
        scratch = torch.empty(len(lengths), 64, 3, dtype=x.dtype, device=x.device)
        indices = torch.arange(len(lengths), dtype=torch.int32, device=x.device)
        fresh = torch.zeros(len(lengths), dtype=torch.bool, device=x.device)
        kwargs = dict(
            query_start_loc=cu,
            has_initial_state=fresh,
            activation="silu",
            metadata=metadata,
        )
        old = causal_conv1d_fn(
            x, w, None, conv_states=old_state, cache_indices=indices + 1, **kwargs
        )
        new = causal_conv1d_fn(
            x,
            w,
            None,
            conv_states=scratch,
            cache_indices=indices,
            null_block_id=None,
            **kwargs,
        )
        pieces = []
        start = 0
        for length in lengths:
            sequence = x[:, start : start + length].float()
            padded = torch.nn.functional.pad(sequence, (3, 0))
            accumulated = torch.zeros_like(sequence)
            for column in range(4):
                accumulated += (
                    padded[:, column : column + length]
                    * w[:, column : column + 1].float()
                )
            pieces.append(torch.nn.functional.silu(accumulated).to(x.dtype))
            start += length
        reference = torch.cat(pieces, dim=1)
        relative = float(
            (new.float() - reference.float()).norm()
            / reference.float().norm().clamp_min(1e-12)
        )
        equal = torch.equal(old, new) and bool(torch.isfinite(new).all())
        report["checks"].append(
            {
                "stage": "convolution",
                "lengths": lengths,
                "bitwise_equal": equal,
                "poisoned_initial_state": True,
                "independent_reference_relative_l2": relative,
            }
        )
        save()
        if not equal or relative > 0.01:
            raise ValueError("scratch convolution reads continuation state")
    for lengths in ([1, 3, 127, 129], [17, 65, 513], [32768]):
        counts = [(n + 15) // 16 for n in lengths]
        cache = torch.randn(
            sum(counts), 2, 4, 16, 256, dtype=torch.bfloat16, device="cuda"
        )
        table = torch.zeros(len(lengths), max(counts), dtype=torch.int32, device="cuda")
        q = torch.randn(sum(lengths), 16, 256, dtype=torch.bfloat16, device="cuda")
        cursor = 0
        keys, values = [], []
        for i, (n, pages) in enumerate(zip(lengths, counts, strict=True)):
            table[i, :pages] = torch.arange(cursor, cursor + pages, device="cuda")
            for which, parts in ((0, keys), (1, values)):
                parts.append(
                    cache[cursor : cursor + pages, which]
                    .permute(0, 2, 1, 3)
                    .flatten(0, 1)[:n]
                )
            cursor += pages
        k, v = torch.cat(keys), torch.cat(values)
        cu = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()],
            dtype=torch.int32,
            device="cuda",
        )
        cp = torch.tensor(
            [0, *torch.tensor(counts).cumsum(0).tolist()],
            dtype=torch.int32,
            device="cuda",
        )
        old, new = torch.empty_like(q), torch.empty_like(q)
        paged_forward(
            query=q,
            kv_cache=cache,
            block_tables=table,
            cum_seq_lens_q=cu,
            cum_seq_lens_kv=cp,
            seq_lens=torch.tensor(lengths, dtype=torch.int32, device="cuda"),
            max_q_len=max(lengths),
            max_kv_len=max(lengths),
            batch_size=len(lengths),
            out=old,
        )
        packed_forward(q, k, v, cu, max(lengths), new)
        equal = torch.equal(old, new) and bool(torch.isfinite(new).all())
        report["checks"].append(
            {"stage": "attention", "lengths": lengths, "bitwise_equal": equal}
        )
        save()
        if not equal:
            raise ValueError("fresh K/V differs from the identical paged arithmetic")
    for lengths in ([1, 3, 127, 129], [17, 65, 513]):
        n = sum(lengths)
        q = torch.randn(n, 4, 128, dtype=torch.bfloat16, device="cuda")
        k, v = torch.randn_like(q), torch.randn(n, 8, 128, dtype=q.dtype, device="cuda")
        # Match fused_post_conv_prep: real serving Q/K are L2-normalized.
        q = torch.nn.functional.normalize(q.float(), dim=-1).to(q.dtype)
        k = torch.nn.functional.normalize(k.float(), dim=-1).to(k.dtype)
        g = torch.rand(n, 8, device="cuda") * 0.9
        beta = torch.rand_like(g)
        cu = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()],
            dtype=torch.int32,
            device="cuda",
        )
        kwargs = dict(q=q, k=k, v=v, g=g, beta=beta, cu_seqlens=cu)
        old, _ = chunk_gated_delta_rule(
            **kwargs,
            initial_state=torch.zeros(len(lengths), 8, 128, 128, device="cuda"),
            output_final_state=True,
        )
        new = chunk_gated_delta_rule(
            **kwargs, initial_state=None, output_final_state=False
        )
        equal = torch.equal(old, new) and bool(torch.isfinite(new).all())
        report["checks"].append(
            {
                "stage": "gdn",
                "lengths": lengths,
                "bitwise_equal": equal,
                "reference_finite": bool(torch.isfinite(old).all()),
                "candidate_finite": bool(torch.isfinite(new).all()),
                "relative_l2": float(
                    (old.float() - new.float()).norm()
                    / old.float().norm().clamp_min(1e-12)
                ),
            }
        )
        save()
        if not equal:
            raise ValueError("discard-state GDN differs from zero-state arithmetic")
    report["passed"] = True
    save()
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
