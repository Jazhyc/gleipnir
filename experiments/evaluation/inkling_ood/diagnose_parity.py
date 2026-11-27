#!/usr/bin/env python3
"""Diagnose the failed batched-versus-singleton canary: with prefix caching
off, score the 13 canary prompts as a batch twice and as singletons twice.
Numerical consistency only; no labels are read and nothing is selected."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.evaluation.inkling_ood.interface import render_margin_prompt
from experiments.training.tool_trajectory_monitoring.benchmark_gpt_oss_ood import (
    binary_logprobs,
)
from gleipnir.binary_evaluation import binary_token_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix-caching", action="store_true")
    parser.add_argument("--config", default="inkling_effort_none")
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()
    config = json.loads(
        Path(f"experiments/evaluation/inkling_ood/{args.config}.json").read_text()
    )
    canary_ids = json.loads(
        Path(f"results/inkling_ood/{args.config}/canary_result.json").read_text()
    )["canary_ids"]
    wanted = set(canary_ids)
    prompts_by_id = {}
    with Path(config["scope"]["input"]).open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["id"] in wanted:
                prompts_by_id[row["id"]] = row["prompt"]

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model, engine = config["model"], config["engine"]
    tokenizer = AutoTokenizer.from_pretrained(model["id"], revision=model["revision"])
    token_ids = binary_token_ids(tokenizer)
    effort = config["prompt"]["reasoning_effort"]
    prompts = [
        render_margin_prompt(tokenizer, prompts_by_id[i], reasoning_effort=effort)
        for i in canary_ids
    ]
    batched_tokens = args.max_num_batched_tokens or int(
        engine["max_num_batched_tokens"]
    )
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
        limit_mm_per_prompt=dict(engine["limit_mm_per_prompt"]),
        max_num_seqs=int(engine["max_num_seqs"]),
        max_num_batched_tokens=batched_tokens,
        enable_prefix_caching=args.prefix_caching,
        enforce_eager=args.enforce_eager,
        seed=int(engine["seed"]),
    )

    def margins(batch: list[str]) -> list[float]:
        result = []
        for output in llm.generate(batch, sampling):
            logprob_0, logprob_1 = binary_logprobs(output, token_ids)
            result.append(logprob_1 - logprob_0)
        return result

    passes = {
        "batch_a": margins(prompts),
        "batch_b": margins(prompts),
        "single_a": [margins([p])[0] for p in prompts],
        "single_b": [margins([p])[0] for p in prompts],
        "batch_reversed": margins(prompts[::-1])[::-1],
    }
    names = list(passes)
    pairs = {
        f"{a}__{b}": {
            "mean_abs_margin_difference": sum(
                abs(x - y) for x, y in zip(passes[a], passes[b], strict=True)
            )
            / len(prompts),
            "max_abs_margin_difference": max(
                abs(x - y) for x, y in zip(passes[a], passes[b], strict=True)
            ),
        }
        for i, a in enumerate(names)
        for b in names[i + 1 :]
    }
    report = {
        "prefix_caching": args.prefix_caching,
        "max_num_batched_tokens": batched_tokens,
        "enforce_eager": args.enforce_eager,
        "canary_ids": canary_ids,
        "margins_1_minus_0": passes,
        "pairs": pairs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for name, pair in pairs.items():
        print(name, {k: round(v, 4) for k, v in pair.items()}, flush=True)
    print("DIAGNOSE_DONE", flush=True)


if __name__ == "__main__":
    main()
