#!/usr/bin/env python3
"""Split Flint's OOD scoring time into tokenization and engine time.

Gigatoken 0.10.0 and 0.11.0 cannot load Flint's tokenizer (unsupported Split
pre-tokenizer), so this measures the ceiling a faster tokenizer could reach:
the full suite is scored from token IDs, with the stock tokenizer timed
separately in the caller. Scores must equal the main run's."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from experiments.flint_ood.interface import render_margin_prompt
from experiments.tool_trajectory_monitoring.benchmark_gpt_oss_ood import (
    batches,
    binary_logprobs,
)
from gleipnir.binary_evaluation import binary_token_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(Path("experiments/flint_ood/flint.json").read_text())
    with Path(config["scope"]["input"]).open() as handle:
        rows = [json.loads(line) for line in handle]
    with (args.reference / "predictions.jsonl").open() as handle:
        reference = {row["id"]: row for row in map(json.loads, handle)}
    main_result = json.loads((args.reference / "result.json").read_text())

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model, engine = config["model"], config["engine"]
    tokenizer = AutoTokenizer.from_pretrained(model["id"], revision=model["revision"])
    token_ids = binary_token_ids(tokenizer)
    started = time.perf_counter()
    texts = [render_margin_prompt(tokenizer, str(row["prompt"])) for row in rows]
    render_seconds = time.perf_counter() - started
    sampling = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=len(token_ids),
        logprob_token_ids=token_ids,
        allowed_token_ids=token_ids,
    )
    llm = LLM(
        model=model["id"],
        tokenizer=model["id"],
        revision=model["revision"],
        tokenizer_revision=model["revision"],
        dtype=model["dtype"],
        tensor_parallel_size=1,
        gpu_memory_utilization=float(engine["gpu_memory_utilization"]),
        max_model_len=int(engine["max_model_len"]),
        block_size=int(engine["block_size"]),
        max_num_seqs=int(engine["max_num_seqs"]),
        max_num_batched_tokens=int(engine["max_num_batched_tokens"]),
        enable_prefix_caching=bool(engine["enable_prefix_caching"]),
        seed=int(engine["seed"]),
    )
    encode_seconds = generate_seconds = 0.0
    tokens = 0
    worst = 0.0
    indices = list(range(len(rows)))
    for number, batch in enumerate(batches(indices, int(engine["batch_rows"])), 1):
        before = time.perf_counter()
        encoded = [tokenizer.encode(texts[i], add_special_tokens=False) for i in batch]
        middle = time.perf_counter()
        outputs = llm.generate([{"prompt_token_ids": ids} for ids in encoded], sampling)
        after = time.perf_counter()
        encode_seconds += middle - before
        generate_seconds += after - middle
        for i, ids, output in zip(batch, encoded, outputs, strict=True):
            tokens += len(ids)
            logprob_0, logprob_1 = binary_logprobs(output, token_ids)
            expected = reference[rows[i]["id"]]
            if len(ids) != expected["prompt_tokens"]:
                raise RuntimeError(f"token count differs for id={rows[i]['id']!r}")
            worst = max(
                worst,
                abs((logprob_1 - logprob_0) - expected["logit_margin_1_minus_0"]),
            )
        if number % 10 == 0:
            print(
                f"batch={number} encode={encode_seconds:.0f}s "
                f"generate={generate_seconds:.0f}s",
                flush=True,
            )
    main_seconds = main_result["runtime"]["full_suite_seconds_this_invocation"]
    report = {
        "rows": len(rows),
        "prompt_tokens": tokens,
        "chat_template_render_seconds": render_seconds,
        "stock_tokenizer_encode_seconds_single_thread": encode_seconds,
        "stock_tokenizer_tokens_per_second": tokens / encode_seconds,
        "engine_seconds_from_token_ids": generate_seconds,
        "rows_per_second_from_token_ids": len(rows) / generate_seconds,
        "prompt_tokens_per_second_from_token_ids": tokens / generate_seconds,
        "main_run_full_suite_seconds_text_prompts": main_seconds,
        "main_run_rows_scored": main_result["runtime"]["rows_scored_this_invocation"],
        "max_abs_margin_difference_from_main_run": worst,
        "gigatoken": "0.10.0 and 0.11.0 cannot load Flint's Split pre-tokenizer",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    print("THROUGHPUT_DONE", flush=True)


if __name__ == "__main__":
    main()
