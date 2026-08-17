#!/usr/bin/env python3
"""Generate reasoning, then score terminal binary logits for AQ SFT adapters."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.data_scaling.prepare import read_jsonl  # noqa: E402
from experiments.reasoning_sft_scaling.core import (  # noqa: E402
    CANONICAL_PREFIX,
    VISION_EXCLUDE_PATTERN,
    output_text,
    prefix_before_prediction,
    reasoning_student_prompt,
)
from gleipnir.binary_evaluation import (  # noqa: E402
    balanced_smoke_records,
    binary_token_ids,
    metric_views,
    score_from_output,
)
from gleipnir.qwen35_adapter_rebase import sha256_file  # noqa: E402


def validate_adapter(job: dict[str, Any]) -> None:
    from safetensors import safe_open

    adapter_dir = Path(job["adapter_dir"])
    weights = adapter_dir / "adapter_model.safetensors"
    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    if config.get("exclude_modules") != VISION_EXCLUDE_PATTERN:
        raise ValueError(f"{adapter_dir} does not exclude visual modules")
    if sha256_file(weights) != job["serving_weights_sha256"]:
        raise ValueError(f"serving weight checksum mismatch: {adapter_dir}")
    with safe_open(weights, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
    if len(keys) != int(job["tensor_count"]) or any(
        not key.startswith(CANONICAL_PREFIX) for key in keys
    ):
        raise ValueError(f"noncanonical serving adapter: {adapter_dir}")


def render_prompts(tokenizer: Any, records: list[dict[str, Any]]) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": reasoning_student_prompt(record["student_prompt"]),
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for record in records
    ]


def evaluate_one(
    llm: Any,
    generation_sampling: Any,
    margin_sampling: Any,
    prompts: list[str],
    records: list[dict[str, Any]],
    token_ids: list[int],
    output_dir: Path,
    metadata: dict[str, Any],
    request: Any | None,
) -> list[float]:
    started = time.time()
    outputs = llm.generate(prompts, generation_sampling, lora_request=request)
    generation_seconds = time.time() - started
    generations = [output_text(output) for output in outputs]
    parsed = [prefix_before_prediction(text) for text in generations]
    margin_prompts = [
        prompt + prefix
        for prompt, (prefix, _) in zip(prompts, parsed, strict=True)
    ]
    started = time.time()
    margin_outputs = llm.generate(
        margin_prompts, margin_sampling, lora_request=request
    )
    margin_seconds = time.time() - started
    scores = [score_from_output(output, token_ids) for output in margin_outputs]
    frame = pd.DataFrame(
        {
            "dataset": [record["dataset"] for record in records],
            "index": [record["index"] for record in records],
            "label": [int(record["label"]) for record in records],
            "score": scores,
            "generated_prediction": [decision for _, decision in parsed],
            "parse_error": [decision is None for _, decision in parsed],
            "generation": generations,
            "generation_tokens": [
                0 if not output.outputs else len(output.outputs[0].token_ids)
                for output in outputs
            ],
            "finish_reason": [
                None if not output.outputs else str(output.outputs[0].finish_reason)
                for output in outputs
            ],
            "prompt_sha256": [
                hashlib.sha256(prompt.encode()).hexdigest() for prompt in prompts
            ],
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_json(
        output_dir / "predictions.jsonl",
        orient="records",
        lines=True,
        double_precision=15,
    )
    result = {
        **metadata,
        "protocol": (
            "greedy reasoning generation; remove the terminal generated 0/1; "
            "normalize literal 0/1 next-token logits at that exact boundary"
        ),
        "rows": len(frame),
        "generation_seconds": generation_seconds,
        "margin_seconds": margin_seconds,
        "total_seconds": generation_seconds + margin_seconds,
        "parse_errors": int(frame["parse_error"].sum()),
        "generated_decision_agreement": float(
            (
                frame["generated_prediction"].fillna(-1).astype(int)
                == frame["label"]
            ).mean()
        ),
        "metrics": metric_views(frame[["dataset", "index", "label", "score"]]),
        "score_sha256_float64": hashlib.sha256(
            frame["score"].to_numpy(dtype="float64").tobytes()
        ).hexdigest(),
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    macro = result["metrics"]["macro"]["macro"]
    print(
        f"{metadata['job_name']}: AUROC={macro['auroc']:.6f} "
        f"BA={macro['balanced_accuracy']:.6f} parse={result['parse_errors']} "
        f"seconds={result['total_seconds']:.1f}",
        flush=True,
    )
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jobs", type=Path, default=Path("results/reasoning_sft_scaling/jobs.jsonl")
    )
    parser.add_argument(
        "--validation", type=Path, default=Path("data/data_scaling/validation.jsonl")
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=4608)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--smoke-rows", type=int)
    parser.add_argument("--only-job")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    jobs = read_jsonl(args.jobs.resolve())
    for job in jobs:
        validate_adapter(job)
    if args.only_job:
        jobs = [job for job in jobs if job["job_name"] == args.only_job]
        if len(jobs) != 1:
            raise ValueError(f"expected one job named {args.only_job!r}")
    records = read_jsonl(args.validation.resolve())
    if args.smoke_rows:
        records = balanced_smoke_records(records, args.smoke_rows)
    adapter_dir = Path(jobs[0]["adapter_dir"])
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    prompts = render_prompts(tokenizer, records)
    token_ids = binary_token_ids(tokenizer)
    generation_sampling = SamplingParams(
        max_tokens=args.max_new_tokens, temperature=0.0
    )
    margin_sampling = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=len(token_ids),
        logprob_token_ids=token_ids,
        allowed_token_ids=token_ids,
    )
    llm = LLM(
        model=args.model,
        tokenizer=adapter_dir.as_posix(),
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=True,
        max_lora_rank=16,
        max_model_len=args.max_model_len,
    )
    split_name = "smoke" if args.smoke_rows else "validation"
    campaign_dir = args.jobs.resolve().parent
    base_dir = campaign_dir / "runs" / "base" / split_name
    if args.force or not (base_dir / "result.json").is_file():
        base_scores = evaluate_one(
            llm,
            generation_sampling,
            margin_sampling,
            prompts,
            records,
            token_ids,
            base_dir,
            {"job_name": "base", "fraction": 0.0, "train_rows": 0, "seed": None},
            None,
        )
    else:
        base_scores = pd.read_json(
            base_dir / "predictions.jsonl", lines=True
        )["score"].astype(float).tolist()
    for lora_id, job in enumerate(jobs, start=1):
        output_dir = campaign_dir / "runs" / job["job_name"] / split_name
        if not args.force and (output_dir / "result.json").is_file():
            print(f"skip complete {job['job_name']}", flush=True)
            continue
        scores = evaluate_one(
            llm,
            generation_sampling,
            margin_sampling,
            prompts,
            records,
            token_ids,
            output_dir,
            job,
            LoRARequest(job["job_name"], lora_id, job["adapter_dir"]),
        )
        deltas = [
            abs(left - right)
            for left, right in zip(scores, base_scores, strict=True)
        ]
        maximum = max(deltas)
        effect = {
            "mean_absolute_score_difference": sum(deltas) / len(deltas),
            "maximum_absolute_score_difference": maximum,
            "exact_equal_scores": sum(delta == 0 for delta in deltas),
        }
        (output_dir / "lora_effect.json").write_text(
            json.dumps(effect, indent=2, sort_keys=True) + "\n"
        )
        if maximum <= 1e-6:
            raise RuntimeError(f"LoRA effect fingerprint failed for {job['job_name']}")


if __name__ == "__main__":
    main()
