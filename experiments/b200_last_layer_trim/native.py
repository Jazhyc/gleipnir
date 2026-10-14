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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch

    import gleipnir.serving_mxfp8 as native
    from experiments.b200_context_million.envelope import enable
    from gleipnir.serving.last_query_attention import selected_rows

    torch.manual_seed(97)
    enable(native)
    paths = [
        Path(__file__),
        Path(native.__file__),
        Path("src/gleipnir/serving/last_query_attention.py"),
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
            rows = selected_rows(0, cumulative(queries), len(queries))
            compact = torch.empty(
                (rows.numel(), 16, 256), dtype=torch.bfloat16, device="cuda"
            )
            native.paged_forward(
                query=query.index_select(0, rows),
                cum_seq_lens_q=torch.arange(
                    len(queries) + 1, dtype=torch.int32, device="cuda"
                ),
                max_q_len=1,
                out=compact,
                **common,
            )
            torch.cuda.synchronize()
            expected = full.index_select(0, rows)
            equal = bool(torch.equal(expected, compact))
            case = {
                "name": name,
                "requests": len(queries),
                "query_tokens": sum(queries),
                "max_history_tokens": max(histories),
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
                report["mismatched_cases"].append(name)
            save()
            print("last_query_case", case, flush=True)
            del cache, table, query, full, compact, expected
            torch.cuda.empty_cache()
        report["passed"] = not report["mismatched_cases"]
        save()
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        save()
        raise


if __name__ == "__main__":
    main()
