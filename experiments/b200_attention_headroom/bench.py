"""Isolated MXFP8 D256 attention timing: kernel vs packing, per SM clock cycle."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

QUERY = 32768
HISTORIES = (32768, 65536, 131072, 262144, 524288, 1048576)
HEADS, KV_HEADS, DIM, SMS = 16, 4, 256, 148
# tcgen05 dense FP8 MMA: 8192 FMA per SM per clock (16384 FLOP).
FP8_FLOP_PER_SM_CLOCK = 16384


def causal_pairs(query: int, history: int) -> int:
    """Bottom-right causal query/key pairs for a query chunk ending the history."""
    return query * history - query * (query - 1) // 2


def measure(torch, native, history: int, iterations: int) -> dict:
    pages = (history + 15) // 16
    cache = torch.randn(
        (pages, 2, KV_HEADS, 16, DIM), dtype=torch.bfloat16, device="cuda"
    )
    table = torch.arange(pages, dtype=torch.int32, device="cuda").view(1, -1)
    query = torch.randn((QUERY, HEADS, DIM), dtype=torch.bfloat16, device="cuda")
    q_offsets = torch.tensor([0, QUERY], dtype=torch.int32, device="cuda")
    kwargs = dict(
        query=query,
        kv_cache=cache,
        block_tables=table,
        cum_seq_lens_q=q_offsets,
        cum_seq_lens_kv=torch.tensor([0, pages], dtype=torch.int32, device="cuda"),
        seq_lens=torch.tensor([history], dtype=torch.int32, device="cuda"),
        max_q_len=QUERY,
        max_kv_len=history,
        batch_size=1,
        out=torch.empty_like(query),
    )
    key, value = native.split_paged_cache(cache)
    offsets = torch.tensor([0, history], dtype=torch.int32, device="cuda")

    def pack():
        native.produce(query, q_offsets, QUERY)
        native.produce(key, offsets, history, table=table, total=history)
        native.produce(value, offsets, history, table=table, total=history, column=True)

    def forward():
        native.paged_forward(**kwargs)

    def timed(fn) -> float:
        fn()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        values = []
        for _ in range(iterations):
            start.record()
            fn()
            end.record()
            end.synchronize()
            values.append(start.elapsed_time(end))
        return statistics.median(values)

    sampler = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=clocks.sm,power.draw",
            "--format=csv,noheader,nounits",
            "-lms=100",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    time.sleep(0.3)
    total_ms = timed(forward)
    pack_ms = timed(pack)
    sampler.terminate()
    samples = [
        line.split(", ") for line in sampler.communicate()[0].strip().splitlines()
    ]
    clocks = sorted(int(s[0]) for s in samples if s and s[0].isdigit())
    power = max(float(s[1]) for s in samples if len(s) > 1)
    clock = statistics.median(clocks) * 1e6
    kernel_ms = total_ms - pack_ms
    rate = 4 * HEADS * DIM * causal_pairs(QUERY, history) / (kernel_ms / 1000)
    return {
        "history": history,
        "total_ms": total_ms,
        "pack_ms": pack_ms,
        "kernel_ms": kernel_ms,
        "pack_share": pack_ms / total_ms,
        "kernel_pflops": rate / 1e15,
        "median_sm_clock_mhz": clock / 1e6,
        "max_power_w": power,
        "fraction_of_clock_adjusted_fp8_peak": rate
        / (FP8_FLOP_PER_SM_CLOCK * SMS * clock),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    import torch

    import gleipnir.serving_mxfp8 as native
    from experiments.b200_context_million.envelope import enable

    enable(native)
    torch.manual_seed(3)
    report = {"gpu": torch.cuda.get_device_name(), "cases": []}
    for history in HISTORIES:
        case = measure(torch, native, history, args.iterations)
        report["cases"].append(case)
        print("attention_headroom", json.dumps(case), flush=True)
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
