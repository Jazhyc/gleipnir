#!/usr/bin/env python3
"""Evaluate capacity-scaling LoRAs or one full-finetuned model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.adapter_capacity_scaling.prepare import read_jsonl  # noqa: E402
from experiments.adapter_capacity_scaling.run_train import (  # noqa: E402
    completed_model,
)
from gleipnir.binary_evaluation import (  # noqa: E402
    binary_token_ids,
    evaluate_vllm_one,
)


def prompts_for(tokenizer: Any, records: list[dict[str, Any]]) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": record["student_prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        + "Prediction:"
        for record in records
    ]


def job_metadata(job: dict[str, Any]) -> dict[str, Any]:
    training_metadata = json.loads(
        (Path(job["model_dir"]) / "training_metadata.json").read_text()
    )
    rebase_path = Path(job["model_dir"]) / "rebase_manifest.json"
    rebase = json.loads(rebase_path.read_text()) if rebase_path.is_file() else None
    return {
        "job_name": job["job_name"],
        "capacity_kind": job["capacity_kind"],
        "seed": int(job["seed"]),
        "rank": job["rank"],
        "lora_alpha": job["lora_alpha"],
        "train_rows": int(job["train_rows"]),
        "model_dir": job["model_dir"],
        "causal_adapter_dir": job.get("causal_adapter_dir"),
        "adapter_rebase_source_sha256": (
            None if rebase is None else rebase["source_sha256"]
        ),
        "adapter_rebase_destination_sha256": (
            None if rebase is None else rebase["destination_sha256"]
        ),
        **training_metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jobs",
        type=Path,
        default=Path("results/adapter_capacity_scaling_causal/lambda_jobs.jsonl"),
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("data/data_scaling/validation.jsonl"),
    )
    parser.add_argument("--mode", choices=("lora", "full"), required=True)
    parser.add_argument("--job-name")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--max-model-len", type=int, default=4608)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    jobs_path = args.jobs.resolve()
    jobs = read_jsonl(jobs_path)
    selected = [job for job in jobs if job["capacity_kind"] == args.mode]
    if args.job_name is not None:
        selected = [job for job in selected if job["job_name"] == args.job_name]
    if not selected:
        raise ValueError("no jobs matched the requested evaluation mode")
    if args.mode == "full" and len(selected) != 1:
        raise ValueError("full evaluation requires exactly one --job-name")
    incomplete = [job["job_name"] for job in selected if not completed_model(job)]
    if incomplete:
        raise FileNotFoundError(f"incomplete trained models: {incomplete[:3]}")

    records = read_jsonl(args.validation.resolve())
    tokenizer_path = selected[0]["model_dir"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    prompts = prompts_for(tokenizer, records)
    token_ids = binary_token_ids(tokenizer)
    sampling = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=len(token_ids),
        logprob_token_ids=token_ids,
        allowed_token_ids=token_ids,
    )

    if args.mode == "lora":
        llm = LLM(
            model=args.model,
            tokenizer=tokenizer_path,
            dtype="bfloat16",
            tensor_parallel_size=1,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_lora=True,
            max_lora_rank=max(int(job["rank"]) for job in selected),
            max_model_len=args.max_model_len,
        )
        base_dir = jobs_path.parent / "runs" / "base" / "validation"
        evaluate_vllm_one(
            llm,
            sampling,
            prompts,
            records,
            token_ids,
            base_dir,
            {"job_name": "base", "capacity_kind": "base", "seed": None},
            None,
        )
        for lora_id, job in enumerate(selected, start=1):
            evaluate_vllm_one(
                llm,
                sampling,
                prompts,
                records,
                token_ids,
                Path(job["output_dir"]) / "validation",
                job_metadata(job),
                LoRARequest(job["job_name"], lora_id, job["model_dir"]),
            )
    else:
        job = selected[0]
        llm = LLM(
            model=job["model_dir"],
            tokenizer=job["model_dir"],
            dtype="bfloat16",
            tensor_parallel_size=1,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
        )
        evaluate_vllm_one(
            llm,
            sampling,
            prompts,
            records,
            token_ids,
            Path(job["output_dir"]) / "validation",
            job_metadata(job),
            None,
        )


if __name__ == "__main__":
    main()
