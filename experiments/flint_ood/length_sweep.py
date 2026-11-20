#!/usr/bin/env python3
"""Single-request latency by prompt length for Flint BF16, on the synthetic
exact-length prompts of the serving length sweep
(`experiments/serving_concurrency/lengths.py`, `native` phase: 256 to 262,144
tokens, one excluded warmup, then five requests, three from 131,072).

The Qwen sweeps time requests through the HTTP scoring server. vLLM 0.34 has
no such server for Flint here, so requests go to the in-process engine: once
as text (tokenization included, as in the Qwen numbers, but no HTTP) and once
as token IDs (engine only). Scores are timing payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

from gleipnir.binary_evaluation import binary_token_ids

NATIVE = 262144
LENGTHS = [2**i for i in range(8, 19)]


def synthetic(tokenizer, length: int) -> dict:
    """Exact-length plain text; a copy of `b200_long_context.run.synthetic`,
    whose module imports the HTTP benchmark stack."""
    body = "".join(
        f"Benchmark record {i}: amber cedar river glass, a neutral observation.\n"
        for i in range(256)
    )
    text = body * (length // 1024 + 2)
    for _ in range(8):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == length:
            break
        text = tokenizer.decode(ids[:length])
    if len(tokenizer.encode(text, add_special_tokens=False)) != length:
        raise ValueError("synthetic prompt length did not round-trip")
    return {
        "prompt": text,
        "prompt_tokens": length,
        "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def repeats(length: int) -> int:
    return 5 if length < 131072 else 3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(Path("experiments/flint_ood/flint.json").read_text())

    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model, engine = config["model"], config["engine"]
    tokenizer = AutoTokenizer.from_pretrained(model["id"], revision=model["revision"])
    token_ids = binary_token_ids(tokenizer)
    # Leave room for the generated token at the native limit.
    rows = [synthetic(tokenizer, n if n < NATIVE else NATIVE - 1) for n in LENGTHS]
    sampling = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=len(token_ids),
        logprob_token_ids=token_ids,
        allowed_token_ids=token_ids,
    )
    started = time.time()
    llm = LLM(
        model=model["id"],
        tokenizer=model["id"],
        revision=model["revision"],
        tokenizer_revision=model["revision"],
        dtype=model["dtype"],
        tensor_parallel_size=1,
        gpu_memory_utilization=float(engine["gpu_memory_utilization"]),
        max_model_len=NATIVE,
        block_size=int(engine["block_size"]),
        max_num_seqs=int(engine["max_num_seqs"]),
        max_num_batched_tokens=int(engine["max_num_batched_tokens"]),
        enable_prefix_caching=False,
        seed=int(engine["seed"]),
    )
    report = {
        "model": model,
        "vllm_version": vllm.__version__,
        "gpu_name": torch.cuda.get_device_name(0),
        "engine_initialization_seconds": time.time() - started,
        "max_model_len": NATIVE,
        "max_num_batched_tokens": int(engine["max_num_batched_tokens"]),
        "prefix_caching": False,
        "measurements": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for row in rows:
        length = row["prompt_tokens"]
        ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
        record = {"prompt_tokens": length, "prompt_sha256": row["prompt_sha256"]}
        try:
            for mode, prompt in (
                ("text", row["prompt"]),
                ("token_ids", {"prompt_token_ids": ids}),
            ):
                llm.generate([prompt], sampling)  # excluded warmup
                latencies = []
                for _ in range(repeats(length)):
                    before = time.perf_counter()
                    output = llm.generate([prompt], sampling)[0]
                    latencies.append(time.perf_counter() - before)
                    if len(output.prompt_token_ids) != length:
                        raise ValueError("engine token count differs")
                median = statistics.median(latencies)
                record[mode] = {
                    "latency_seconds": latencies,
                    "median_ms": 1000 * median,
                    "prompt_tokens_per_second": length / median,
                }
                print(
                    f"length {length} {mode}: {1000 * median:.1f} ms "
                    f"{length / median:,.0f} tok/s",
                    flush=True,
                )
        except Exception as error:  # record the failed length and stop
            record["error"] = repr(error)[:500]
            report["measurements"].append(record)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(f"length {length} failed: {record['error']}", flush=True)
            break
        report["measurements"].append(record)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("LENGTH_SWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
