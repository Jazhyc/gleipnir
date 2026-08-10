#!/usr/bin/env python3
"""Evaluate all scaling adapters with direct constrained 0/1 logits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.data_scaling.prepare import read_jsonl  # noqa: E402
from gleipnir.binary_evaluation import (  # noqa: E402
    balanced_smoke_records,
    binary_token_ids,
    evaluate_vllm_one,
)


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
        base_scores = evaluate_vllm_one(
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
        adapter_scores = evaluate_vllm_one(
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
