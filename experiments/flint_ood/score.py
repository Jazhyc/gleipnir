#!/usr/bin/env python3
"""Score Flint's direct binary logits on the frozen OOD suite (vLLM >= 0.33)."""

from __future__ import annotations

import argparse
import hashlib
import math
import time
from pathlib import Path
from typing import Any

from experiments.flint_ood.interface import MODEL_ID, REVISION, render_margin_prompt
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
from gleipnir.binary_evaluation import binary_token_ids, logprob_value
from gleipnir.qwen35_adapter_rebase import sha256_file

DEFAULT_CONFIG = Path("experiments/flint_ood/flint.json")
DEFAULT_OUTPUT = Path("results/flint_ood/flint")


def validate_config(config: dict[str, Any]) -> None:
    if config.get("status") != "frozen_evaluation":
        raise ValueError("benchmark config must be frozen before evaluation")
    model = config.get("model", {})
    if (model.get("id"), model.get("revision")) != (MODEL_ID, REVISION):
        raise ValueError("frozen model must be the pinned Flint revision")
    if model.get("dtype") != "bfloat16":
        raise ValueError("Flint benchmark uses the upstream BF16 weights")
    prompt = config.get("prompt", {})
    if prompt.get("role") != "teacher":
        raise ValueError("benchmark must use the full teacher prompt")
    if prompt.get("assistant_suffix") != "<|assistant|>":
        raise ValueError("frozen Flint assistant boundary differs")
    if prompt.get("decision_prefix") != "Prediction:":
        raise ValueError("frozen decision prefix must be Prediction:")
    if config.get("scoring", {}).get("tokens") != ["0", "1"]:
        raise ValueError("frozen decision tokens must be literal 0 and 1")
    engine = config.get("engine", {})
    if int(engine.get("audited_max_prompt_tokens", 0)) >= int(
        engine.get("max_model_len", 0)
    ):
        raise ValueError("max_model_len does not cover the audited prompt maximum")
    if int(engine.get("tensor_parallel_size", 0)) != 1:
        raise ValueError("Flint's kernels are validated on one GPU only")


def opening_token_diagnostic(
    llm: Any, tokenizer: Any, prompts: list[str], sampling_cls: Any
) -> list[dict[str, Any]]:
    """Unconstrained first assistant token: would Flint have reasoned first?"""
    sampling = sampling_cls(max_tokens=1, temperature=0.0, logprobs=20)
    rows = []
    for output in llm.generate(prompts, sampling):
        top = output.outputs[0].logprobs[0] or {}
        ranked = sorted(
            ((int(k), logprob_value(v)) for k, v in top.items()),
            key=lambda item: -item[1],
        )
        rows.append(
            {
                "top": [
                    {"token": tokenizer.convert_ids_to_tokens(i), "logprob": lp}
                    for i, lp in ranked[:5]
                ],
                "first_token": tokenizer.convert_ids_to_tokens(ranked[0][0]),
            }
        )
    return rows


def run(args: argparse.Namespace) -> None:
    config = load_json(args.config)
    validate_config(config)
    records = validate_inputs(config)
    config_sha256 = sha256_file(args.config)
    template = load_prompt_set().teacher
    if template.template_sha256 != config["prompt"]["template_sha256"]:
        raise ValueError("working-tree prompt differs from the frozen config")

    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model, engine, prompt_config = config["model"], config["engine"], config["prompt"]
    if vllm.__version__ != engine["vllm_version"]:
        raise RuntimeError(f"vLLM {vllm.__version__} differs from the frozen version")
    tokenizer = AutoTokenizer.from_pretrained(model["id"], revision=model["revision"])
    token_ids = binary_token_ids(tokenizer)

    def render(row: dict[str, Any]) -> str:
        return render_margin_prompt(
            tokenizer,
            str(row["prompt"]),
            assistant_suffix=prompt_config["assistant_suffix"],
            decision_prefix=prompt_config["decision_prefix"],
        )

    # Frozen rendering and length audit, before the engine is built.
    digest = hashlib.sha256()
    lengths = []
    for row in records:
        text = render(row)
        digest.update(text.encode() + b"\0")
        lengths.append(len(tokenizer.encode(text, add_special_tokens=False)))
    if (
        digest.hexdigest() != prompt_config["rendered_sha256"]
        or max(lengths) != engine["audited_max_prompt_tokens"]
        or sum(lengths) != engine["audited_total_prompt_tokens"]
    ):
        raise RuntimeError("Flint rendering or token audit differs from frozen config")

    sampling = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=len(token_ids),
        logprob_token_ids=token_ids,
        allowed_token_ids=token_ids,
    )
    engine_started = time.time()
    llm = LLM(
        model=model["id"],
        tokenizer=model["id"],
        revision=model["revision"],
        tokenizer_revision=model["revision"],
        dtype=model["dtype"],
        tensor_parallel_size=1,
        gpu_memory_utilization=float(engine["gpu_memory_utilization"]),
        max_model_len=int(engine["max_model_len"]),
        # Flint's decode kernel needs 16-token KV blocks; vLLM 0.34 defaults to 64.
        block_size=int(engine["block_size"]),
        max_num_seqs=int(engine["max_num_seqs"]),
        max_num_batched_tokens=int(engine["max_num_batched_tokens"]),
        enable_prefix_caching=bool(engine["enable_prefix_caching"]),
        seed=int(engine["seed"]),
    )
    engine_seconds = time.time() - engine_started
    runtime = {
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        "gpu_name": torch.cuda.get_device_name(0),
        "engine_initialization_seconds": engine_seconds,
    }
    atomic_write_json(
        args.output / "engine_ready.json", {"config_sha256": config_sha256, **runtime}
    )

    predictions_path = args.output / "predictions.jsonl"
    existing = (
        []
        if args.force or not predictions_path.is_file()
        else load_jsonl(predictions_path)
    )
    completed = validate_existing(existing, records, config_sha256)

    def score_records(
        batch: list[dict[str, Any]], *, persist: bool = True
    ) -> list[dict[str, Any]]:
        prompts = [render(row) for row in batch]
        # The template adds no BOS; keep the tokenizer from adding specials.
        outputs = llm.generate(prompts, sampling)
        if len(outputs) != len(batch):
            raise RuntimeError("vLLM output count differs from request count")
        packed_rows = []
        for record, margin_prompt, output in zip(batch, prompts, outputs, strict=True):
            packed = prediction_row(
                record,
                margin_prompt,
                output,
                token_ids,
                config_sha256,
                int(engine["max_model_len"]),
            )
            expected = lengths[index_by_id[str(record["id"])]]
            if packed["prompt_tokens"] != expected:
                raise RuntimeError(
                    f"engine tokenized id={record['id']!r} to "
                    f"{packed['prompt_tokens']} tokens, audit says {expected}"
                )
            packed_rows.append(packed)
            if persist:
                completed[packed["id"]] = packed
        if persist:
            atomic_write_jsonl(
                predictions_path, ordered_predictions(records, completed)
            )
        return packed_rows

    index_by_id = {str(row["id"]): i for i, row in enumerate(records)}
    canary = balanced_canary_rows(
        records,
        list(config["scope"]["sources"]),
        rows_per_source_label=int(engine["canary_rows_per_source_label"]),
    )
    longest = records[max(range(len(records)), key=lengths.__getitem__)]
    if engine.get("canary_include_longest") and longest["id"] not in {
        row["id"] for row in canary
    }:
        canary.append(longest)
    pending_canary = [row for row in canary if str(row["id"]) not in completed]
    canary_started = time.time()
    if pending_canary:
        score_records(pending_canary)
    canary_seconds = time.time() - canary_started
    canary_predictions = [completed[str(row["id"])] for row in canary]
    if not all(math.isfinite(float(row["score"])) for row in canary_predictions):
        raise RuntimeError("balanced backend canary failed")
    repeats = [score_records([row], persist=False)[0] for row in canary]
    differences = [
        abs(float(a["score"]) - float(b["score"]))
        for a, b in zip(canary_predictions, repeats, strict=True)
    ]
    parity = {
        "mean_absolute_error": sum(differences) / len(differences),
        "max_absolute_error": max(differences),
    }
    parity["passed"] = (
        parity["mean_absolute_error"] <= engine["canary_singleton_mean_abs_error_max"]
        and parity["max_absolute_error"] <= engine["canary_singleton_max_abs_error_max"]
    )
    opening = opening_token_diagnostic(
        llm,
        tokenizer,
        [render(row)[: -len(prompt_config["decision_prefix"])] for row in canary],
        SamplingParams,
    )
    atomic_write_json(
        args.output / "canary_result.json",
        {
            "campaign_id": config["campaign_id"],
            "config_sha256": config_sha256,
            "purpose": "backend_and_selected_logprob_interface_only",
            "canary_ids": [str(row["id"]) for row in canary],
            "batched_canary_seconds": canary_seconds,
            "batch_singleton_parity": {
                **parity,
                "batched": [row["score"] for row in canary_predictions],
                "singletons": [row["score"] for row in repeats],
            },
            "unconstrained_opening_token": opening,
            **runtime,
            **summarize(canary_predictions),
        },
    )
    if not parity["passed"]:
        raise RuntimeError("batched-versus-singleton canary failed")
    print(f"canary complete rows={len(canary_predictions)} parity={parity}", flush=True)
    if args.stop_after_canary:
        return

    pending = [row for row in records if str(row["id"]) not in completed]
    full_started = time.time()
    for batch_index, batch in enumerate(
        batches(pending, int(engine["batch_rows"])), start=1
    ):
        score_records(batch)
        print(
            f"batch={batch_index} complete={len(completed)}/{len(records)} "
            f"elapsed={time.time() - full_started:.0f}s",
            flush=True,
        )
    full_seconds = time.time() - full_started
    predictions = ordered_predictions(records, completed)
    if len(predictions) != len(records):
        raise RuntimeError("full OOD benchmark is incomplete")
    scored_tokens = sum(lengths[index_by_id[str(row["id"])]] for row in pending)
    result = {
        "campaign_id": config["campaign_id"],
        "config_sha256": config_sha256,
        "model": model,
        "prompt": prompt_config,
        "engine": engine,
        "runtime": {
            **runtime,
            "batched_canary_seconds": canary_seconds,
            "full_suite_seconds_this_invocation": full_seconds,
            "rows_scored_this_invocation": len(pending),
            "prompt_tokens_scored_this_invocation": scored_tokens,
            "rows_per_second": len(pending) / full_seconds if pending else None,
            "prompt_tokens_per_second": scored_tokens / full_seconds
            if pending
            else None,
        },
        "score": "normalized direct probability for literal 1 versus 0",
        **summarize(predictions),
    }
    atomic_write_json(args.output / "result.json", result)
    macro = result["metrics"]["macro"]["macro"]
    print(
        f"complete rows={len(predictions)} "
        f"macro_auroc={macro['auroc']:.6f} "
        f"macro_pauroc_at_20={macro['pauroc_at_20']:.6f} "
        f"full_suite_seconds={full_seconds:.0f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stop-after-canary", action="store_true")
    parser.add_argument("--force", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
