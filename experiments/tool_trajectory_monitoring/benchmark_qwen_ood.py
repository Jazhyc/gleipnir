#!/usr/bin/env python3
"""Benchmark Qwen3.5 direct binary logits on the frozen OOD suite."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

from experiments.tool_trajectory_monitoring.benchmark_gpt_oss_ood import (
    balanced_canary_rows,
    batches,
    ordered_predictions,
    prediction_row,
    summarize,
    validate_existing,
    validate_inputs,
)
from experiments.tool_trajectory_monitoring.prompting import load_prompt_set
from experiments.tool_trajectory_monitoring.qwen_reasoning_core import (
    atomic_write_json,
    atomic_write_jsonl,
    load_json,
    load_jsonl,
)
from gleipnir.binary_evaluation import binary_token_ids
from gleipnir.qwen35_adapter_rebase import sha256_file

DEFAULT_CONFIG = Path(
    "experiments/tool_trajectory_monitoring/qwen_ood_benchmark.json"
)
DEFAULT_OUTPUT = Path("results/tool_trajectory_monitoring/qwen35_9b_teacher_ood")
QWEN_NON_THINKING_ASSISTANT_SUFFIX = (
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)
SUPPORTED_QWEN_MODELS = {
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--stop-after-canary",
        action="store_true",
        help="Validate the backend on the frozen balanced canary, then stop.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("status") != "frozen_evaluation":
        raise ValueError("benchmark config must be frozen before evaluation")
    model_id = config.get("model", {}).get("id")
    if model_id not in SUPPORTED_QWEN_MODELS:
        supported = ", ".join(sorted(SUPPORTED_QWEN_MODELS))
        raise ValueError(f"frozen model must be one of: {supported}")
    prompt = config.get("prompt", {})
    if prompt.get("role") != "teacher":
        raise ValueError("benchmark must use the full teacher prompt")
    if prompt.get("enable_thinking") is not False:
        raise ValueError("direct Qwen benchmark must disable native thinking")
    if prompt.get("assistant_suffix") != QWEN_NON_THINKING_ASSISTANT_SUFFIX:
        raise ValueError("frozen Qwen non-thinking assistant suffix differs")
    if prompt.get("decision_prefix") != "Prediction:":
        raise ValueError("frozen decision prefix must be Prediction:")
    if config.get("scoring", {}).get("tokens") != ["0", "1"]:
        raise ValueError("frozen decision tokens must be literal 0 and 1")
    engine = config.get("engine", {})
    if int(engine.get("audited_max_prompt_tokens", 0)) >= int(
        engine.get("max_model_len", 0)
    ):
        raise ValueError("max_model_len does not cover the audited prompt maximum")
    if int(engine.get("canary_rows_per_source_label", 0)) < 1:
        raise ValueError("canary rows per source-label must be positive")
    if engine.get("language_model_only") is not True:
        raise ValueError("Qwen benchmark must use the text-only language model")


def render_margin_prompt(
    tokenizer: Any,
    user_prompt: str,
    *,
    enable_thinking: bool,
    assistant_suffix: str,
    decision_prefix: str,
) -> str:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    if not rendered.endswith(assistant_suffix):
        raise ValueError(
            "Qwen chat template no longer ends at the frozen non-thinking boundary"
        )
    return f"{rendered}{decision_prefix}"


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    validate_config(config)
    records = validate_inputs(config)
    config_sha256 = sha256_file(args.config)
    prompt_set = load_prompt_set()
    if prompt_set.teacher.template_sha256 != config["prompt"]["template_sha256"]:
        raise ValueError("working-tree teacher prompt differs from the frozen config")

    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model = config["model"]
    engine = config["engine"]
    prompt_config = config["prompt"]
    tokenizer = AutoTokenizer.from_pretrained(
        model["id"], revision=model["revision"]
    )
    token_ids = binary_token_ids(tokenizer)
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
        tensor_parallel_size=int(engine["tensor_parallel_size"]),
        gpu_memory_utilization=float(engine["gpu_memory_utilization"]),
        max_model_len=int(engine["max_model_len"]),
        max_num_seqs=int(engine["max_num_seqs"]),
        max_num_batched_tokens=int(engine["max_num_batched_tokens"]),
        enable_prefix_caching=bool(engine["enable_prefix_caching"]),
        language_model_only=bool(engine["language_model_only"]),
        seed=int(engine["seed"]),
    )

    predictions_path = args.output / "predictions.jsonl"
    existing = (
        []
        if args.force or not predictions_path.is_file()
        else load_jsonl(predictions_path)
    )
    completed = validate_existing(existing, records, config_sha256)

    def score_records(batch: list[dict[str, Any]]) -> None:
        prompts = [
            render_margin_prompt(
                tokenizer,
                str(row["prompt"]),
                enable_thinking=bool(prompt_config["enable_thinking"]),
                assistant_suffix=str(prompt_config["assistant_suffix"]),
                decision_prefix=str(prompt_config["decision_prefix"]),
            )
            for row in batch
        ]
        outputs = llm.generate(prompts, sampling)
        if len(outputs) != len(batch):
            raise RuntimeError("vLLM output count differs from request count")
        for record, margin_prompt, output in zip(batch, prompts, outputs, strict=True):
            packed = prediction_row(
                record,
                margin_prompt,
                output,
                token_ids,
                config_sha256,
                int(engine["max_model_len"]),
            )
            completed[packed["id"]] = packed
        atomic_write_jsonl(predictions_path, ordered_predictions(records, completed))

    canary = balanced_canary_rows(
        records,
        list(config["scope"]["sources"]),
        rows_per_source_label=int(engine["canary_rows_per_source_label"]),
    )
    pending_canary = [row for row in canary if str(row["id"]) not in completed]
    started = time.time()
    if pending_canary:
        score_records(pending_canary)
    canary_predictions = [completed[str(row["id"])] for row in canary]
    if len(canary_predictions) != len(canary) or not all(
        math.isfinite(float(row["score"])) for row in canary_predictions
    ):
        raise RuntimeError("balanced backend canary failed")
    atomic_write_json(
        args.output / "canary_result.json",
        {
            "campaign_id": config["campaign_id"],
            "config_sha256": config_sha256,
            "purpose": "backend_and_selected_logprob_interface_only",
            **summarize(canary_predictions),
        },
    )
    print(f"canary complete rows={len(canary_predictions)}", flush=True)
    if args.stop_after_canary:
        return

    pending = [row for row in records if str(row["id"]) not in completed]
    for batch_index, batch in enumerate(
        batches(pending, int(engine["batch_rows"])), start=1
    ):
        score_records(batch)
        print(
            f"batch={batch_index} complete={len(completed)}/{len(records)}",
            flush=True,
        )
    predictions = ordered_predictions(records, completed)
    if len(predictions) != len(records):
        raise RuntimeError("full OOD benchmark is incomplete")
    result = {
        "campaign_id": config["campaign_id"],
        "config_sha256": config_sha256,
        "model": model,
        "prompt": prompt_config,
        "engine": engine,
        "runtime": {
            "vllm_version": vllm.__version__,
            "torch_version": torch.__version__,
            "gpu_name": torch.cuda.get_device_name(0),
            "elapsed_seconds_this_invocation": time.time() - started,
        },
        "score": "normalized direct probability for literal 1 versus 0",
        **summarize(predictions),
    }
    atomic_write_json(args.output / "result.json", result)
    macro = result["metrics"]["macro"]["macro"]
    pooled = result["metrics"]["pooled"]
    print(
        f"complete rows={len(predictions)} "
        f"macro_auroc={macro['auroc']:.6f} "
        f"macro_pauroc_at_20={macro['pauroc_at_20']:.6f} "
        f"pooled_auroc={pooled['auroc']:.6f} "
        f"pooled_pauroc_at_20={pooled['pauroc_at_20']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
