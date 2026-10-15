"""Require bitwise-equal MXFP8 rows for full-chunk and last-query-only attention."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

CASES = (
    ("first_chunk", [32768], [32768]),
    ("second_chunk", [32768], [65536]),
    ("chunk_1m_history", [32768], [1048576]),
    ("irregular", [17], [32785]),
    ("mixed_prefills", [100, 32, 4000, 1], [100, 5000, 4000, 777]),
    (
        "batch_128",
        [1 + i % 3 * 50 for i in range(128)],
        [51 + i * 37 for i in range(128)],
    ),
)
# Multi-request cases also attend only every other request, as when only some
# prompts finish in a step, using the serving metadata-selection helper.
SUBSETS = ("mixed_prefills", "batch_128")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from vllm.v1.attention.backends.flashinfer import TRTLLMPrefill

    import gleipnir.serving_mxfp8 as native
    from experiments.b200_context_million.envelope import enable
    from gleipnir.serving.last_query_attention import select_prefills, selected_rows

    torch.manual_seed(97)
    enable(native)
    paths = [
        Path(__file__),
        Path(native.__file__),
        Path("src/gleipnir/serving/last_query_attention.py"),
        Path("src/gleipnir/serving/last_query_tail.py"),
    ]
    report = {
        "passed": False,
        "gpu": torch.cuda.get_device_name(),
        "bitwise_cases": 0,
        "mismatched_cases": [],
        "cases": [],
        "sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    }

    def save() -> None:
        report["sha256"] = hashlib.sha256(
            json.dumps(report["sources"], sort_keys=True).encode()
        ).hexdigest()
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def cumulative(values: list[int]) -> torch.Tensor:
        return torch.tensor(
            [0, *torch.tensor(values).cumsum(0).tolist()],
            dtype=torch.int32,
            device="cuda",
        )

    save()
    try:
        for name, queries, histories in CASES:
            pages = [(n + 15) // 16 for n in histories]
            padding = sum(pages)
            cache = torch.randn(
                (padding + sum(pages), 2, 4, 16, 256),
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
                (sum(queries), 16, 256), dtype=torch.bfloat16, device="cuda"
            )
            common = dict(
                kv_cache=cache,
                block_tables=table,
                cum_seq_lens_kv=cumulative(pages),
                seq_lens=torch.tensor(histories, dtype=torch.int32, device="cuda"),
                max_kv_len=max(histories),
                batch_size=len(queries),
            )
            full = torch.empty_like(query)
            native.paged_forward(
                query=query,
                cum_seq_lens_q=cumulative(queries),
                max_q_len=max(queries),
                out=full,
                **common,
            )
            all_rows = selected_rows(0, cumulative(queries), len(queries))
            prefill = TRTLLMPrefill(
                block_tables=table,
                seq_lens=common["seq_lens"],
                cum_seq_lens_q=cumulative(queries),
                cum_seq_lens_kv=common["cum_seq_lens_kv"],
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
                    (rows.numel(), 16, 256), dtype=torch.bfloat16, device="cuda"
                )
                native.paged_forward(
                    query=query.index_select(0, rows),
                    kv_cache=cache,
                    block_tables=selected.block_tables,
                    cum_seq_lens_q=selected.cum_seq_lens_q,
                    cum_seq_lens_kv=selected.cum_seq_lens_kv,
                    seq_lens=selected.seq_lens,
                    max_q_len=1,
                    max_kv_len=selected.max_seq_len,
                    batch_size=len(chosen),
                    out=compact,
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
