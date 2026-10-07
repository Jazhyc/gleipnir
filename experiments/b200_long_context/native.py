"""Validate an extended descriptor against frozen short and long-history arithmetic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.b200_long_context.envelope import LIMIT, enable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch

    import gleipnir.serving_mxfp8 as native
    from experiments.b200_attention_gdn_serving.mxfp8_reference import attention

    torch.manual_seed(71)
    torch.backends.cuda.matmul.allow_tf32 = False
    paths = [
        Path(__file__),
        Path(__file__).with_name("envelope.py"),
        Path(native.__file__),
        Path("experiments/b200_attention_gdn_serving/mxfp8_reference.py"),
    ]
    report = {
        "passed": False,
        "context_limit": LIMIT,
        "short_bitwise_cases": 0,
        "gpu": torch.cuda.get_device_name(),
        "checks": [],
        "sources": {
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
    }

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def case(queries, histories):
        batch = len(queries)
        pages = [(n + 15) // 16 for n in histories]
        cache = torch.randn(
            (sum(pages), 2, 4, 16, 256), dtype=torch.bfloat16, device="cuda"
        )
        table = torch.zeros((batch, max(pages)), dtype=torch.int32, device="cuda")
        offset = 0
        for row, count in enumerate(pages):
            table[row, :count] = torch.arange(
                offset, offset + count, device="cuda"
            ).flip(0)
            offset += count
        q = torch.randn((sum(queries), 16, 256), dtype=torch.bfloat16, device="cuda")
        output = torch.empty_like(q)

        def cumulative(xs):
            return torch.tensor(
                [0, *torch.tensor(xs).cumsum(0).tolist()],
                dtype=torch.int32,
                device="cuda",
            )

        kwargs = dict(
            query=q,
            kv_cache=cache,
            block_tables=table,
            cum_seq_lens_q=cumulative(queries),
            cum_seq_lens_kv=cumulative(pages),
            seq_lens=torch.tensor(histories, dtype=torch.int32, device="cuda"),
            max_q_len=max(queries),
            max_kv_len=max(histories),
            batch_size=batch,
            out=output,
        )
        return kwargs

    save()
    try:
        short = [
            case([17], [32768]),
            case(
                [1 + i % 3 for i in range(128)], [33 + i % 4 * 16 for i in range(128)]
            ),
        ]
        expected = []
        for kwargs in short:
            native.paged_forward(**kwargs)
            expected.append(kwargs["out"].clone())
        torch.cuda.synchronize()
        enable(native)
        for kwargs, old in zip(short, expected, strict=True):
            native.paged_forward(**kwargs)
            torch.cuda.synchronize()
            if not torch.equal(old, kwargs["out"]):
                raise ValueError("extended descriptor changed short/batched outputs")
            report["short_bitwise_cases"] += 1
        del short, expected, kwargs, old
        for queries, histories in (
            ([17], [32785]),
            *[([32768], [n]) for n in (65536, 131072, 262144)],
        ):
            kwargs = case(queries, histories)
            native.paged_forward(**kwargs)
            torch.cuda.synchronize()
            q, cache, table, observed = (
                kwargs[k] for k in ("query", "kv_cache", "block_tables", "out")
            )
            qn, kn = queries[0], histories[0]
            key, value = [
                cache[:, which][table[0].long()].permute(0, 2, 1, 3).flatten(0, 1)[:kn]
                for which in (0, 1)
            ]
            if qn == 17:
                oracle = attention(q, key, value)
                selected = observed.float()
            else:
                # Both causal history boundaries align to V's 32-token scale
                # groups, so prefix-only independent quantization is identical.
                indexes = (31, qn - 1)
                oracle = torch.cat(
                    [
                        attention(
                            q[i : i + 1],
                            key[: kn - qn + i + 1],
                            value[: kn - qn + i + 1],
                        )
                        for i in indexes
                    ]
                )
                selected = observed[list(indexes)].float()
            error = float((selected - oracle).norm() / oracle.norm())
            finite = bool(torch.isfinite(observed).all())
            positions = (
                torch.arange(qn, device="cuda")
                if qn == 17
                else torch.tensor(indexes, device="cuda")
            )
            sampled_q = q[positions]
            bf16_heads = []
            for head in range(16):
                scores = sampled_q[:, head].float() @ key[:, head // 4].float().T / 16
                mask = torch.arange(kn, device="cuda")[None] > (
                    kn - qn + positions[:, None]
                )
                scores.masked_fill_(mask, float("-inf"))
                bf16_heads.append(scores.softmax(-1) @ value[:, head // 4].float())
            bf16_oracle = torch.stack(bf16_heads, dim=1)
            bf16_error = float((selected - bf16_oracle).norm() / bf16_oracle.norm())
            check = {
                "query_tokens": qn,
                "history_tokens": kn,
                "finite": finite,
                "quantized_relative_l2": error,
                "bf16_relative_l2": bf16_error,
                "strict_bf16_passed": bf16_error <= 0.05,
                "sampled_query_positions": "all" if qn == 17 else [31, qn - 1],
            }
            report["checks"].append(check)
            save()
            print("native_long_context", check, flush=True)
            if not finite or error > 0.01:
                raise ValueError("extended context native arithmetic failed")
            del (
                kwargs,
                q,
                cache,
                table,
                observed,
                key,
                value,
                oracle,
                selected,
                sampled_q,
                bf16_heads,
                bf16_oracle,
                scores,
                mask,
            )
        report["passed"] = True
        save()
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        save()
        raise


if __name__ == "__main__":
    main()
