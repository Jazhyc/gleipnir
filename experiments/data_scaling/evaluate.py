#!/usr/bin/env python3
"""Evaluate all scaling adapters with direct constrained 0/1 logits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.data_scaling.prepare import read_jsonl  # noqa: E402
from gleipnir.metrics import evaluate_binary_monitor  # noqa: E402


def binary_token_ids(tokenizer: Any) -> list[int]:
    ids = []
    for value in ("0", "1"):
        encoded = tokenizer.encode(value, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"{value!r} tokenized as {encoded}, expected one token")
        ids.append(int(encoded[0]))
    if len(set(ids)) != 2:
        raise ValueError(f"binary token ids are not distinct: {ids}")
    return ids


def logprob_value(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float(value.logprob if hasattr(value, "logprob") else value["logprob"])


def score_from_output(output: Any, token_ids: list[int]) -> float:
    if not output.outputs or not output.outputs[0].logprobs:
        raise RuntimeError("vLLM returned no first-token logprobs")
    values = output.outputs[0].logprobs[0] or {}
    expanded = {int(key): logprob_value(value) for key, value in values.items()}
    missing = [token_id for token_id in token_ids if token_id not in expanded]
    if missing:
        raise RuntimeError(f"vLLM omitted requested token logprobs: {missing}")
    difference = max(-80.0, min(80.0, expanded[token_ids[1]] - expanded[token_ids[0]]))
    return 1.0 / (1.0 + math.exp(-difference))


def metric_views(frame: pd.DataFrame) -> dict[str, Any]:
    macro = evaluate_binary_monitor(frame)
    pooled_frame = frame.assign(_pool="all")
    pooled = evaluate_binary_monitor(pooled_frame, group_column="_pool")["groups"][0]
    scenarios = {}
    scenario_patterns = (
        ("instructed", "dev-instructed-deception"),
        ("varied", "dev-varied-deception"),
    )
    for name, pattern in scenario_patterns:
        subset = frame[frame["dataset"].str.contains(pattern, regex=False)]
        if not subset.empty:
            scenarios[name] = evaluate_binary_monitor(subset)
    return {"macro": macro, "pooled": pooled, "scenarios": scenarios}


def balanced_smoke_records(
    records: list[dict[str, Any]], limit: int
) -> list[dict[str, Any]]:
    """Select a small single-dataset slice containing both binary labels."""
    if limit < 2:
        raise ValueError("--smoke-rows must be at least 2")
    by_dataset: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for record in records:
        label = int(record["label"])
        by_dataset.setdefault(str(record["dataset"]), {0: [], 1: []})[label].append(
            record
        )
    for groups in by_dataset.values():
        per_label = min(len(groups[0]), len(groups[1]), limit // 2)
        if per_label:
            return groups[0][:per_label] + groups[1][:per_label]
    raise ValueError("validation data has no dataset containing both labels")


def evaluate_one(
    llm: Any,
    sampling: Any,
    prompts: list[str],
    records: list[dict[str, Any]],
    token_ids: list[int],
    output_dir: Path,
    metadata: dict[str, Any],
    request: Any | None,
) -> list[float]:
    started = time.time()
    outputs = llm.generate(prompts, sampling, lora_request=request)
    elapsed = time.time() - started
    scores = [score_from_output(output, token_ids) for output in outputs]
    frame = pd.DataFrame(
        {
            "dataset": [record["dataset"] for record in records],
            "index": [record["index"] for record in records],
            "label": [int(record["label"]) for record in records],
            "score": scores,
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
        "score": "normalized direct next-token probability for literal 1 vs 0",
        "threshold": 0.5,
        "rows": len(frame),
        "seconds": elapsed,
        "rows_per_second": len(frame) / elapsed,
        "metrics": metric_views(frame),
        "score_sha256_float64": hashlib.sha256(
            frame["score"].to_numpy(dtype="float64").tobytes()
        ).hexdigest(),
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    primary = result["metrics"]["macro"]["macro"]
    print(
        f"{metadata['job_name']}: macro_auroc={primary['auroc']:.6f} "
        f"macro_ba={primary['balanced_accuracy']:.6f} seconds={elapsed:.1f}",
        flush=True,
    )
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jobs", type=Path, default=Path("results/data_scaling/jobs.jsonl")
    )
    parser.add_argument(
        "--validation", type=Path, default=Path("data/data_scaling/validation.jsonl")
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--max-model-len", type=int, default=4608)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument(
        "--smoke-rows",
        type=int,
        default=None,
        help="Evaluate the first N rows and require the selected LoRA to move scores.",
    )
    parser.add_argument(
        "--only-job",
        help="Evaluate one adapter job (intended for the preflight smoke test).",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    jobs = read_jsonl(args.jobs.resolve())
    incomplete = [
        job["job_name"]
        for job in jobs
        if not (
            Path(job["adapter_dir"]) / "adapter_model.safetensors"
        ).is_file()
    ]
    if incomplete:
        raise FileNotFoundError(
            f"{len(incomplete)} adapters are incomplete: {incomplete[:3]}"
        )
    records = read_jsonl(args.validation.resolve())
    if args.smoke_rows is not None:
        records = balanced_smoke_records(records, args.smoke_rows)
    selected_jobs = jobs
    if args.only_job is not None:
        selected_jobs = [job for job in jobs if job["job_name"] == args.only_job]
        if len(selected_jobs) != 1:
            raise ValueError(f"expected exactly one job named {args.only_job!r}")
    validation_subdir = (
        "validation_smoke" if args.smoke_rows is not None else "validation"
    )
    tokenizer = AutoTokenizer.from_pretrained(jobs[0]["adapter_dir"])
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": record["student_prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        + "Prediction:"
        for record in records
    ]
    token_ids = binary_token_ids(tokenizer)
    sampling = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=len(token_ids),
        logprob_token_ids=token_ids,
        allowed_token_ids=token_ids,
    )
    llm = LLM(
        model=args.model,
        tokenizer=jobs[0]["adapter_dir"],
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=True,
        max_lora_rank=16,
        max_model_len=args.max_model_len,
    )
    base_scores = None
    if not args.skip_base:
        base_dir = args.jobs.resolve().parent / "runs" / "base" / validation_subdir
        base_scores = evaluate_one(
            llm,
            sampling,
            prompts,
            records,
            token_ids,
            base_dir,
            {"job_name": "base", "seed": None, "fraction": 0.0, "train_rows": 0},
            None,
        )
    for lora_id, job in enumerate(selected_jobs, start=1):
        request = LoRARequest(job["job_name"], lora_id, job["adapter_dir"])
        adapter_scores = evaluate_one(
            llm,
            sampling,
            prompts,
            records,
            token_ids,
            Path(job["output_dir"]) / validation_subdir,
            {
                "job_name": job["job_name"],
                "seed": job["seed"],
                "fraction": job["fraction"],
                "train_rows": job["train_rows"],
                "adapter_dir": job["adapter_dir"],
                **{
                    key: job[key]
                    for key in (
                        "source_adapter_dir",
                        "adapter_rebase_manifest",
                        "adapter_rebase_source_sha256",
                        "adapter_rebase_destination_sha256",
                    )
                    if key in job
                },
            },
            request,
        )
        if args.smoke_rows is not None:
            if base_scores is None:
                raise ValueError("smoke evaluation requires the base-model comparison")
            max_delta = max(
                abs(adapter - base)
                for adapter, base in zip(adapter_scores, base_scores, strict=True)
            )
            if max_delta <= 1e-6:
                raise RuntimeError(
                    f"LoRA smoke test failed: maximum score delta was {max_delta:.3g}"
                )
            print(f"LoRA smoke test passed: max_score_delta={max_delta:.8f}")


if __name__ == "__main__":
    main()
