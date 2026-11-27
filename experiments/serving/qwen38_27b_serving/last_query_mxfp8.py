"""Require bitwise-equal MXFP8 rows for full-chunk and last-query attention.

The config-derived sibling of ``experiments/b200_last_layer_trim/native.py``
(which fixes GQA 16/4): the same cases and request subsets, run through
``gleipnir.serving.mxfp8_geometry.paged_forward`` at the served model's head
counts, with the 1M-history envelope enabled as in the 4B receipt. The
receipt binds one attention shape and ``precision: mxfp8``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PAGE = 16


def attention_shape(model: Path) -> dict:
    config = json.loads((model / "config.json").read_text())
    text = config.get("text_config", config)
    return {
        "query_heads": int(text["num_attention_heads"]),
        "kv_heads": int(text["num_key_value_heads"]),
        "head_dim": int(text["head_dim"]),
        "page_size": PAGE,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from vllm.v1.attention.backends.flashinfer import TRTLLMPrefill

    from experiments.b200_context_million.envelope import enable
    from experiments.b200_last_layer_trim.native import CASES, SUBSETS
    from gleipnir.serving import mxfp8
    from gleipnir.serving.last_query_attention import select_prefills, selected_rows
    from gleipnir.serving.mxfp8_geometry import paged_forward

    torch.manual_seed(97)
    shape = attention_shape(args.model)
    heads, kv_heads, dim = shape["query_heads"], shape["kv_heads"], shape["head_dim"]
    if dim != 256:
        raise ValueError("MXFP8 serving attention requires head dimension 256")
    enable(mxfp8)
    paths = [
        Path(__file__),
        Path("src/gleipnir/serving/mxfp8.py"),
        Path("src/gleipnir/serving/mxfp8_geometry.py"),
        Path("src/gleipnir/serving/last_query_attention.py"),
        Path("src/gleipnir/serving/last_query_tail.py"),
    ]
    report = {
        "passed": False,
        "precision": "mxfp8",
        "kernel": "gleipnir.serving.mxfp8_geometry.paged_forward",
        "attention_shape": shape,
        "gpu": torch.cuda.get_device_name(),
        "bitwise_cases": 0,
        "mismatched_cases": [],
        "cases": [],
        "sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    }

    def save() -> None:
        report["sha256"] = hashlib.sha256(
            json.dumps(
                {"sources": report["sources"], "shape": shape}, sort_keys=True
            ).encode()
        ).hexdigest()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def cumulative(values: list[int]) -> torch.Tensor:
        return torch.tensor(
            [0, *torch.tensor(values).cumsum(0).tolist()],
            dtype=torch.int32,
            device="cuda",
        )

    def attend(query, cache, table, seq_lens, cum_q, cum_kv, max_q, max_kv, out):
        paged_forward(
            query_heads=heads,
            kv_heads=kv_heads,
            query=query,
            kv_cache=cache,
            block_tables=table,
            seq_lens=seq_lens,
            max_q_len=max_q,
            max_kv_len=max_kv,
            batch_size=seq_lens.numel(),
            cum_seq_lens_q=cum_q,
            cum_seq_lens_kv=cum_kv,
            out=out,
        )

    save()
    try:
        for name, queries, histories in CASES:
            pages = [(n + PAGE - 1) // PAGE for n in histories]
            padding = sum(pages)
            cache = torch.randn(
                (padding + sum(pages), 2, kv_heads, PAGE, dim),
                dtype=torch.bfloat16,
                device="cuda",
            )
            table = torch.zeros(
                (len(pages), max(pages)), dtype=torch.int32, device="cuda"
            )
            offset = padding
            for row, count in enumerate(pages):
                table[row, :count] = torch.arange(offset, offset + count).flip(0)
                offset += count
            query = torch.randn(
                (sum(queries), heads, dim), dtype=torch.bfloat16, device="cuda"
            )
            seq_lens = torch.tensor(histories, dtype=torch.int32, device="cuda")
            full = torch.empty_like(query)
            attend(
                query,
                cache,
                table,
                seq_lens,
                cumulative(queries),
                cumulative(pages),
                max(queries),
                max(histories),
                full,
            )
            all_rows = selected_rows(0, cumulative(queries), len(queries))
            prefill = TRTLLMPrefill(
                block_tables=table,
                seq_lens=seq_lens,
                cum_seq_lens_q=cumulative(queries),
                cum_seq_lens_kv=cumulative(pages),
                max_q_len=max(queries),
                max_seq_len=max(histories),
            )
            variants = [(name, list(range(len(queries))))]
            if name in SUBSETS:
                variants.append((f"{name}_subset", list(range(1, len(queries), 2))))
            for label, chosen in variants:
                indices = torch.tensor(chosen, dtype=torch.int64, device="cuda")
                rows = all_rows.index_select(0, indices)
                selected = select_prefills(
                    prefill, indices, max(histories[i] for i in chosen)
                )
                compact = torch.empty(
                    (rows.numel(), heads, dim), dtype=torch.bfloat16, device="cuda"
                )
                attend(
                    query.index_select(0, rows),
                    cache,
                    selected.block_tables,
                    selected.seq_lens,
                    selected.cum_seq_lens_q,
                    selected.cum_seq_lens_kv,
                    1,
                    selected.max_seq_len,
                    compact,
                )
                torch.cuda.synchronize()
                expected = full.index_select(0, rows)
                equal = bool(torch.equal(expected, compact))
                case = {
                    "name": label,
                    "requests": len(chosen),
                    "query_tokens": sum(queries[i] for i in chosen),
                    "max_history_tokens": selected.max_seq_len,
                    "bitwise_equal": equal,
                    "finite": bool(torch.isfinite(compact).all()),
                    "max_absolute_difference": float(
                        (expected.float() - compact.float()).abs().max()
                    ),
                }
                report["cases"].append(case)
                if equal and case["finite"]:
                    report["bitwise_cases"] += 1
                else:
                    report["mismatched_cases"].append(label)
                save()
                print("last_query_case", case, flush=True)
            del cache, table, query, full, compact, expected, prefill, selected
            torch.cuda.empty_cache()
        report["passed"] = not report["mismatched_cases"]
        save()
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        save()
        raise


if __name__ == "__main__":
    main()
