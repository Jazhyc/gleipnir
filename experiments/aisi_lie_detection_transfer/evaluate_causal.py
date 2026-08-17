#!/usr/bin/env python3
"""Evaluate frozen base and adapter targets on the AISI paper testbeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.data_scaling.prepare import read_jsonl  # noqa: E402
from experiments.deception_distillation.train_student_sft import (  # noqa: E402
    gated_delta_kernel_modules,
)
from experiments.training_procedure_screen.evaluate_causal import (  # noqa: E402
    score_adapter,
)
from gleipnir.binary_evaluation import (  # noqa: E402
    balanced_smoke_records,
    binary_token_ids,
)
from gleipnir.metrics import evaluate_binary_monitor  # noqa: E402


def metric_views(frame: pd.DataFrame) -> dict[str, Any]:
    """Report subject-macro, testbed-macro, pooled, and overlap diagnostics."""
    subject_macro = evaluate_binary_monitor(frame)
    pooled = evaluate_binary_monitor(frame.assign(_pool="all"), group_column="_pool")[
        "groups"
    ][0]
    testbeds = {
        str(testbed): evaluate_binary_monitor(subset)
        for testbed, subset in frame.groupby("testbed", sort=True)
    }
    overlap = {}
    for seen, subset in frame.groupby("seen_internal_user", sort=True):
        if set(subset["label"]) == {0, 1}:
            overlap[str(bool(seen)).lower()] = evaluate_binary_monitor(
                subset.assign(_pool="all"), group_column="_pool"
            )["groups"][0]
    return {
        "subject_macro": subject_macro,
        "testbeds": testbeds,
        "pooled": pooled,
        "seen_internal_user": overlap,
    }


def write_result(
    output_dir: Path,
    records: list[dict[str, Any]],
    scores: list[float],
    metadata: dict[str, Any],
    seconds: float,
    score_description: str = (
        "causal selected-position probability for literal 1 versus 0"
    ),
) -> None:
    frame = pd.DataFrame(
        {
            "dataset": [record["dataset"] for record in records],
            "index": [record["index"] for record in records],
            "label": [int(record["label"]) for record in records],
            "testbed": [record["testbed"] for record in records],
            "subject_model": [record["subject_model"] for record in records],
            "seen_internal_user": [
                bool(record["seen_internal_user"]) for record in records
            ],
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
        "score": score_description,
        "threshold": 0.5,
        "rows": len(frame),
        "seconds": seconds,
        "rows_per_second": len(frame) / seconds,
        "metrics": metric_views(frame),
        "score_sha256_float64": hashlib.sha256(
            frame["score"].to_numpy(dtype="float64").tobytes()
        ).hexdigest(),
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )


def strength_jobs(jobs_path: Path, strength_id: str) -> list[dict[str, Any]]:
    jobs = [
        job
        for job in read_jsonl(jobs_path)
        if str(job.get("strength_id")) == strength_id
    ]
    jobs.sort(key=lambda job: int(job["seed"]))
    if [int(job["seed"]) for job in jobs] != [0, 1, 2]:
        raise ValueError(f"{strength_id}: expected exactly seeds 0, 1, and 2")
    for job in jobs:
        adapter = Path(job["causal_adapter_dir"])
        if not (adapter / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(f"missing adapter for {job['job_name']}: {adapter}")
    return jobs


def soft_only_jobs(jobs_path: Path) -> list[dict[str, Any]]:
    """Select the three original soft-only baseline adapters."""
    jobs = [
        job
        for job in read_jsonl(jobs_path)
        if str(job.get("job_name", "")).startswith("baseline-seed")
    ]
    jobs.sort(key=lambda job: int(job["seed"]))
    if [int(job["seed"]) for job in jobs] != [0, 1, 2]:
        raise ValueError("soft-only: expected exactly baseline seeds 0, 1, and 2")
    for job in jobs:
        if (
            str(job.get("intervention_family")) != "baseline"
            or float(job.get("soft_loss_weight", -1.0)) != 1.0
            or float(job.get("direct_loss_weight", -1.0)) != 0.0
        ):
            raise ValueError(f"soft-only recipe mismatch for {job['job_name']}")
        adapter = Path(job["causal_adapter_dir"])
        if not (adapter / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(f"missing adapter for {job['job_name']}: {adapter}")
    return jobs


def target_jobs(
    jobs_path: Path, soft_jobs_path: Path, strength_id: str
) -> list[dict[str, Any]]:
    """Resolve a frozen evaluation target to its training manifest."""
    if strength_id == "soft-only":
        return soft_only_jobs(soft_jobs_path)
    return strength_jobs(jobs_path, strength_id)


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
        "--soft-jobs",
        type=Path,
        default=Path("results/training_procedure_screen/jobs.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/aisi_lie_detection_transfer/runs"),
    )
    parser.add_argument("--include-base", action="store_true")
    parser.add_argument(
        "--strength-id",
        action="append",
        choices=("soft-only", "0500", "hard-only"),
        default=[],
    )
    parser.add_argument("--seed", action="append", type=int, choices=(0, 1, 2))
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4608)
    parser.add_argument("--smoke-rows", type=int)
    args = parser.parse_args()
    if not args.include_base and not args.strength_id:
        parser.error("select --include-base and/or at least one --strength-id")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils.import_utils import is_flash_linear_attention_available

    records = read_jsonl(args.evaluation.resolve())
    if args.smoke_rows is not None:
        records = balanced_smoke_records(records, args.smoke_rows)
    source_manifest = json.loads(args.manifest.resolve().read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
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
    tokenized = [
        tokenizer.encode(prompt, add_special_tokens=False)[-args.max_length :]
        for prompt in prompts
    ]
    if not is_flash_linear_attention_available():
        raise RuntimeError("external causal evaluation requires pinned FLA kernels")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": torch.cuda.current_device()},
    )
    kernels = gated_delta_kernel_modules(model)
    if not kernels or any(not name.startswith("fla.ops.") for name in kernels):
        raise RuntimeError(f"causal evaluation did not bind FLA kernels: {kernels}")
    token_ids = binary_token_ids(tokenizer)
    common_metadata = {
        "model": args.model,
        "evaluation_base": "bf16",
        "flash_linear_attention": kernels,
        "source_dataset": source_manifest["source_dataset"],
        "source_revision": source_manifest["source_revision"],
        "evaluation_sha256": source_manifest["evaluation_sha256"],
        "selection_rule": source_manifest["selection_rule"],
    }
    if args.include_base:
        scores, seconds = score_adapter(
            model,
            tokenizer,
            tokenized,
            token_ids,
            batch_size=args.batch_size,
            decision_head_mode="token_logits",
        )
        output = args.output_dir.resolve() / "base" / "base"
        write_result(
            output,
            records,
            scores,
            {
                **common_metadata,
                "target": "base",
                "seed": None,
                "adapter_path": None,
            },
            seconds,
        )
        metric = json.loads((output / "result.json").read_text())["metrics"][
            "subject_macro"
        ]["macro"]
        print(
            f"base: auroc={metric['auroc']:.6f} "
            f"ba={metric['balanced_accuracy']:.6f} seconds={seconds:.1f}",
            flush=True,
        )
    selected = [
        (strength_id, job)
        for strength_id in args.strength_id
        for job in target_jobs(
            args.jobs.resolve(), args.soft_jobs.resolve(), strength_id
        )
        if not args.seed or int(job["seed"]) in args.seed
    ]
    if not selected:
        return
    first_strength, first_job = selected[0]
    first_name = f"{first_strength}-seed{int(first_job['seed'])}"
    model = PeftModel.from_pretrained(
        model,
        str(first_job["causal_adapter_dir"]),
        adapter_name=first_name,
        is_trainable=False,
    )
    for strength_id, job in selected[1:]:
        adapter_name = f"{strength_id}-seed{int(job['seed'])}"
        model.load_adapter(
            str(job["causal_adapter_dir"]),
            adapter_name=adapter_name,
            is_trainable=False,
        )
    kernels = gated_delta_kernel_modules(model)
    if not kernels or any(not name.startswith("fla.ops.") for name in kernels):
        raise RuntimeError(f"adapter evaluation did not bind FLA kernels: {kernels}")
    for strength_id, job in selected:
        seed = int(job["seed"])
        adapter_name = f"{strength_id}-seed{seed}"
        model.set_adapter(adapter_name)
        started = time.time()
        scores, scoring_seconds = score_adapter(
            model,
            tokenizer,
            tokenized,
            token_ids,
            batch_size=args.batch_size,
            decision_head_mode="token_logits",
        )
        seconds = time.time() - started
        if abs(seconds - scoring_seconds) > 1.0:
            raise RuntimeError("inconsistent evaluation timing")
        output = args.output_dir.resolve() / strength_id / f"seed{seed}"
        write_result(
            output,
            records,
            scores,
            {
                **common_metadata,
                "target": strength_id,
                "seed": seed,
                "adapter_path": str(job["causal_adapter_dir"]),
                "training_job": job["job_name"],
            },
            seconds,
        )
        metric = json.loads((output / "result.json").read_text())["metrics"][
            "subject_macro"
        ]["macro"]
        print(
            f"{adapter_name}: auroc={metric['auroc']:.6f} "
            f"ba={metric['balanced_accuracy']:.6f} seconds={seconds:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
