#!/usr/bin/env python3
"""Continuously batched vLLM evaluation of frozen external-transfer targets."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.aisi_lie_detection_transfer.evaluate_causal import (  # noqa: E402
    strength_jobs,
    write_result,
)
from experiments.data_scaling.prepare import read_jsonl  # noqa: E402
from gleipnir.binary_evaluation import (  # noqa: E402
    balanced_smoke_records,
    binary_token_ids,
    score_from_output,
)


def select_jobs(
    jobs_path: Path, strength_ids: list[str], seeds: list[int] | None = None
) -> list[tuple[str, dict[str, Any]]]:
    """Select frozen strength/seed targets in a stable order."""
    allowed = set(seeds or [])
    return [
        (strength_id, job)
        for strength_id in strength_ids
        for job in strength_jobs(jobs_path, strength_id)
        if not allowed or int(job["seed"]) in allowed
    ]


def prompts_for(tokenizer: Any, records: list[dict[str, Any]]) -> list[str]:
    """Render exactly the causal evaluator's direct-boundary prompts."""
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


def score_target(
    llm: Any,
    sampling: Any,
    prompts: list[str],
    token_ids: list[int],
    request: Any | None,
) -> tuple[list[float], float]:
    """Score one frozen base/LoRA target through vLLM's continuous scheduler."""
    started = time.time()
    outputs = llm.generate(prompts, sampling, lora_request=request)
    seconds = time.time() - started
    return [score_from_output(output, token_ids) for output in outputs], seconds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=Path("data/aisi_lie_detection_transfer/evaluation.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/aisi_lie_detection_transfer/manifest.json"),
    )
    parser.add_argument(
        "--jobs",
        type=Path,
        default=Path(
            "results/training_procedure_screen/hard_label_strength/jobs.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/aisi_lie_detection_transfer/vllm_runs"),
    )
    parser.add_argument("--include-base", action="store_true")
    parser.add_argument(
        "--strength-id", action="append", choices=("0500", "hard-only"), default=[]
    )
    parser.add_argument("--seed", action="append", type=int, choices=(0, 1, 2))
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--max-model-len", type=int, default=4608)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--smoke-rows", type=int)
    args = parser.parse_args()
    if not args.include_base and not args.strength_id:
        parser.error("select --include-base and/or at least one --strength-id")

    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    jobs_path = args.jobs.resolve()
    selected = select_jobs(jobs_path, args.strength_id, args.seed)
    for _, job in selected:
        adapter = Path(job["model_dir"])
        if not (adapter / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(f"missing serving adapter: {adapter}")
        if not (adapter / "rebase_manifest.json").is_file():
            raise FileNotFoundError(f"missing adapter rebase manifest: {adapter}")

    records = read_jsonl(args.evaluation.resolve())
    if args.smoke_rows is not None:
        records = balanced_smoke_records(records, args.smoke_rows)
    source_manifest = json.loads(args.manifest.resolve().read_text())
    tokenizer_path = str(selected[0][1]["model_dir"]) if selected else args.model
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
    llm = LLM(
        model=args.model,
        tokenizer=tokenizer_path,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=bool(selected),
        max_lora_rank=max((int(job["rank"]) for _, job in selected), default=16),
        max_model_len=args.max_model_len,
    )
    common_metadata = {
        "model": args.model,
        "evaluation_base": "bf16",
        "serving_backend": "vllm_continuous_batching",
        "vllm_version": vllm.__version__,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "requested_logprob_token_ids": token_ids,
        "source_dataset": source_manifest["source_dataset"],
        "source_revision": source_manifest["source_revision"],
        "evaluation_sha256": source_manifest["evaluation_sha256"],
        "selection_rule": source_manifest["selection_rule"],
    }
    output_root = args.output_dir.resolve()
    if args.include_base:
        scores, seconds = score_target(llm, sampling, prompts, token_ids, None)
        output = output_root / "base" / "base"
        write_result(
            output,
            records,
            scores,
            {**common_metadata, "target": "base", "seed": None, "adapter_path": None},
            seconds,
            score_description=(
                "vLLM normalized constrained next-token probability for literal 1 "
                "versus 0"
            ),
        )
        print(f"base: rows={len(records)} seconds={seconds:.1f}", flush=True)
    for lora_id, (strength_id, job) in enumerate(selected, start=1):
        seed = int(job["seed"])
        adapter_path = Path(job["model_dir"])
        rebase = json.loads((adapter_path / "rebase_manifest.json").read_text())
        request = LoRARequest(
            f"{strength_id}-seed{seed}", lora_id, adapter_path.as_posix()
        )
        scores, seconds = score_target(llm, sampling, prompts, token_ids, request)
        output = output_root / strength_id / f"seed{seed}"
        write_result(
            output,
            records,
            scores,
            {
                **common_metadata,
                "target": strength_id,
                "seed": seed,
                "adapter_path": adapter_path.as_posix(),
                "training_job": job["job_name"],
                "adapter_rebase_source_sha256": rebase["source_sha256"],
                "adapter_rebase_destination_sha256": rebase["destination_sha256"],
            },
            seconds,
            score_description=(
                "vLLM normalized constrained next-token probability for literal 1 "
                "versus 0"
            ),
        )
        print(
            f"{strength_id}-seed{seed}: rows={len(records)} seconds={seconds:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
