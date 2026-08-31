#!/usr/bin/env python3
"""Benchmark visible non-thinking reasoning followed by Qwen binary logits."""

from __future__ import annotations

import argparse
import hashlib
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.tool_trajectory_monitoring import (
    benchmark_qwen_ordinal_reasoning_ood as ordinal_reasoning,
)
from experiments.tool_trajectory_monitoring.benchmark_gpt_oss_ood import (
    balanced_canary_rows,
    batches,
    ordered_predictions,
    summarize,
    validate_inputs,
)
from experiments.tool_trajectory_monitoring.benchmark_gpt_oss_ood import (
    prediction_row as binary_prediction_row,
)
from experiments.tool_trajectory_monitoring.benchmark_qwen_ood import (
    QWEN_NON_THINKING_ASSISTANT_SUFFIX,
)
from experiments.tool_trajectory_monitoring.benchmark_qwen_ordinal_ood import (
    extract_trajectory,
    render_generation_prompt,
    render_ordinal_user_prompt,
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
    "experiments/tool_trajectory_monitoring/"
    "qwen_binary_reasoning_ood_benchmark.json"
)
DEFAULT_OUTPUT = Path(
    "results/tool_trajectory_monitoring/"
    "qwen35_9b_teacher_binary_reasoning_ood"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stop-after-canary", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("status") != "frozen_evaluation":
        raise ValueError("benchmark config must be frozen before evaluation")
    if config.get("model", {}).get("id") != "Qwen/Qwen3.5-9B":
        raise ValueError("frozen model must be Qwen/Qwen3.5-9B")
    prompt = config.get("prompt", {})
    if prompt.get("role") != "teacher":
        raise ValueError("benchmark must use the full teacher specification")
    if prompt.get("enable_thinking") is not False:
        raise ValueError("visible reasoning must keep native thinking disabled")
    if prompt.get("assistant_suffix") != QWEN_NON_THINKING_ASSISTANT_SUFFIX:
        raise ValueError("frozen Qwen non-thinking assistant suffix differs")
    instruction_path = Path(str(prompt.get("instruction_file", "")))
    if not instruction_path.is_file():
        raise ValueError("binary reasoning instruction file is missing")
    if sha256_file(instruction_path) != prompt.get("instruction_sha256"):
        raise ValueError("binary reasoning instruction hash differs")

    if config.get("scoring", {}).get("tokens") != ["0", "1"]:
        raise ValueError("frozen decision tokens must be literal 0 and 1")
    generation = config.get("generation", {})
    if generation.get("stop") != "\nPrediction:":
        raise ValueError("reasoning must stop at the terminal prediction boundary")
    if generation.get("include_stop_str_in_output") is not True:
        raise ValueError("prediction boundary must remain in the continuation")
    if float(generation.get("temperature", -1)) != 0.0:
        raise ValueError("matched visible reasoning must use greedy decoding")
    max_rationale = int(generation.get("max_tokens", 0))
    if max_rationale < 1:
        raise ValueError("visible reasoning max_tokens must be positive")

    engine = config.get("engine", {})
    audited_max = int(engine.get("audited_max_prompt_tokens", 0))
    audited_total = int(engine.get("audited_total_prompt_tokens", 0))
    max_model_len = int(engine.get("max_model_len", 0))
    if audited_max < 1 or audited_total < audited_max:
        raise ValueError("audited prompt-token accounting is invalid")
    if audited_max + max_rationale + 1 > max_model_len:
        raise ValueError("max_model_len does not cover prompt, rationale, and decision")
    if int(engine.get("canary_rows_per_source_label", 0)) < 1:
        raise ValueError("canary rows per source-label must be positive")
    if engine.get("language_model_only") is not True:
        raise ValueError("Qwen benchmark must use the text-only language model")


def reasoned_prediction_row(
    record: dict[str, Any],
    generation_prompt: str,
    rationale_output: Any,
    margin_prompt: str,
    margin_output: Any,
    tokenizer: Any,
    token_ids: list[int],
    stop: str,
    config_sha256: str,
    max_model_len: int,
) -> dict[str, Any]:
    continuation, rationale_token_ids, boundary_recovery = (
        ordinal_reasoning.reasoning_continuation(rationale_output, stop)
    )
    if margin_prompt != generation_prompt + continuation:
        raise RuntimeError("margin prompt is not the exact reasoning continuation")
    base = binary_prediction_row(
        record,
        margin_prompt,
        margin_output,
        token_ids,
        config_sha256,
        max_model_len,
    )
    initial_prompt_token_ids = list(rationale_output.prompt_token_ids or [])
    margin_prompt_token_ids = list(margin_output.prompt_token_ids or [])
    if not initial_prompt_token_ids or not margin_prompt_token_ids:
        raise RuntimeError("vLLM omitted prompt token IDs")
    generated = margin_output.outputs[0]
    decision_text = str(generated.text)
    decision_token_ids = list(generated.token_ids)
    if decision_text not in {"0", "1"} or decision_token_ids != [
        token_ids[int(decision_text)]
    ]:
        raise RuntimeError("binary decision text or tokenization drifted")
    if len(margin_prompt_token_ids) + 1 > max_model_len:
        raise RuntimeError("reasoning plus decision reaches max_model_len")

    reasoning_context_tokens = len(margin_prompt_token_ids) - len(
        initial_prompt_token_ids
    )
    synthetic_boundary_tokens = (
        len(tokenizer.encode(stop, add_special_tokens=False))
        if boundary_recovery != "none"
        else 0
    )
    logical_output_tokens = len(
        tokenizer.encode(continuation + decision_text, add_special_tokens=False)
    )
    rationale = continuation[: -len(stop)].strip()
    return {
        **base,
        "rationale": rationale,
        "rationale_sha256": hashlib.sha256(rationale.encode("utf-8")).hexdigest(),
        "rationale_token_ids": rationale_token_ids,
        "rationale_tokens": len(rationale_token_ids),
        "reasoning_context_tokens": reasoning_context_tokens,
        "synthetic_boundary_tokens": synthetic_boundary_tokens,
        "rationale_finish_reason": str(rationale_output.outputs[0].finish_reason),
        "rationale_stop_reason": rationale_output.outputs[0].stop_reason,
        "boundary_recovery": boundary_recovery,
        "rationale_truncated": boundary_recovery == "length_cap",
        "contains_think_close": "</think>" in rationale,
        "generated_text": decision_text,
        "generated_token_ids": decision_token_ids,
        "decision_tokens": 1,
        "output_tokens": logical_output_tokens,
        "served_output_tokens": len(rationale_token_ids) + 1,
        "completion_retokenization_delta": logical_output_tokens
        - len(rationale_token_ids)
        - synthetic_boundary_tokens
        - 1,
        "prompt_tokens": len(initial_prompt_token_ids),
        "margin_prompt_tokens": len(margin_prompt_token_ids),
        "generation_prompt_sha256": hashlib.sha256(
            generation_prompt.encode("utf-8")
        ).hexdigest(),
    }


def summarize_reasoning(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize(rows)
    frame = pd.DataFrame(rows)
    summary["output_tokens"] = {
        "total": int(frame["output_tokens"].sum()),
        "min": int(frame["output_tokens"].min()),
        "max": int(frame["output_tokens"].max()),
        "mean": float(frame["output_tokens"].mean()),
    }
    summary["reasoning"] = {
        "tokens_total": int(frame["rationale_tokens"].sum()),
        "tokens_min": int(frame["rationale_tokens"].min()),
        "tokens_max": int(frame["rationale_tokens"].max()),
        "tokens_mean": float(frame["rationale_tokens"].mean()),
        "characters_mean": float(frame["rationale"].str.len().mean()),
        "think_close_rows": int(frame["contains_think_close"].sum()),
        "truncated_rows": int(frame["rationale_truncated"].sum()),
        "boundary_recoveries": dict(Counter(frame["boundary_recovery"])),
        "synthetic_boundary_tokens": int(frame["synthetic_boundary_tokens"].sum()),
        "served_output_tokens": int(frame["served_output_tokens"].sum()),
        "completion_retokenization_delta": int(
            frame["completion_retokenization_delta"].sum()
        ),
        "finish_reasons": dict(Counter(frame["rationale_finish_reason"])),
    }
    summary["margin_prompt_tokens"] = {
        "total": int(frame["margin_prompt_tokens"].sum()),
        "min": int(frame["margin_prompt_tokens"].min()),
        "max": int(frame["margin_prompt_tokens"].max()),
        "mean": float(frame["margin_prompt_tokens"].mean()),
    }
    return summary


def _metric_delta(
    candidate_metrics: dict[str, Any], reference_metrics: dict[str, Any]
) -> dict[str, Any]:
    def view(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, float]:
        return {
            "auroc": float(candidate["auroc"]) - float(reference["auroc"]),
            "pauroc_at_20": float(candidate["pauroc_at_20"])
            - float(reference["pauroc_at_20"]),
        }

    candidate_sources = {
        str(row["group"]): row for row in candidate_metrics["macro"]["groups"]
    }
    reference_sources = {
        str(row["group"]): row for row in reference_metrics["macro"]["groups"]
    }
    if candidate_sources.keys() != reference_sources.keys():
        raise ValueError("paired metric source sets differ")
    return {
        "macro": view(
            candidate_metrics["macro"]["macro"],
            reference_metrics["macro"]["macro"],
        ),
        "pooled": view(candidate_metrics["pooled"], reference_metrics["pooled"]),
        "by_dataset": {
            source: view(candidate_sources[source], reference_sources[source])
            for source in sorted(candidate_sources)
        },
    }


def compare_direct_baseline(
    reasoned_rows: list[dict[str, Any]], direct_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    reasoned_by_id = {str(row["id"]): row for row in reasoned_rows}
    direct_by_id = {str(row["id"]): row for row in direct_rows}
    if len(reasoned_by_id) != len(reasoned_rows) or len(direct_by_id) != len(
        direct_rows
    ):
        raise ValueError("paired comparison contains duplicate IDs")
    if reasoned_by_id.keys() != direct_by_id.keys():
        raise ValueError("reasoned and direct baseline IDs differ")
    for record_id, reasoned in reasoned_by_id.items():
        direct = direct_by_id[record_id]
        if reasoned["source"] != direct["source"] or int(reasoned["label"]) != int(
            direct["label"]
        ):
            raise ValueError(f"paired baseline identity drift for id={record_id!r}")

    reasoned_summary = summarize(reasoned_rows)
    direct_summary = summarize(direct_rows)
    reasoned_scores = np.asarray(
        [float(reasoned_by_id[record_id]["score"]) for record_id in reasoned_by_id]
    )
    direct_scores = np.asarray(
        [float(direct_by_id[record_id]["score"]) for record_id in reasoned_by_id]
    )
    correlation = (
        float(np.corrcoef(reasoned_scores, direct_scores)[0, 1])
        if np.std(reasoned_scores) > 0 and np.std(direct_scores) > 0
        else None
    )
    return {
        "direct_metrics": direct_summary["metrics"],
        "delta_reasoned_minus_direct": _metric_delta(
            reasoned_summary["metrics"], direct_summary["metrics"]
        ),
        "decision_agreement": float(
            np.mean((reasoned_scores >= 0.5) == (direct_scores >= 0.5))
        ),
        "mean_absolute_score_delta": float(
            np.mean(np.abs(reasoned_scores - direct_scores))
        ),
        "pearson_score_correlation": correlation,
    }


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    validate_config(config)
    records = validate_inputs(
        {
            **config,
            "prompt": {
                "prompt_set_id": config["prompt"]["binary_prompt_set_id"],
                "template_sha256": config["prompt"]["binary_template_sha256"],
            },
        }
    )
    config_sha256 = sha256_file(args.config)
    binary_template = load_prompt_set().teacher
    if binary_template.template_sha256 != config["prompt"]["binary_template_sha256"]:
        raise ValueError("working-tree binary teacher prompt differs")
    instruction = Path(config["prompt"]["instruction_file"]).read_text(
        encoding="utf-8"
    )

    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model = config["model"]
    engine = config["engine"]
    prompt_config = config["prompt"]
    generation = config["generation"]
    tokenizer = AutoTokenizer.from_pretrained(
        model["id"], revision=model["revision"]
    )
    token_ids = binary_token_ids(tokenizer)
    reasoning_sampling = SamplingParams(
        max_tokens=int(generation["max_tokens"]),
        temperature=float(generation["temperature"]),
        stop=[str(generation["stop"])],
        include_stop_str_in_output=bool(generation["include_stop_str_in_output"]),
    )
    margin_sampling = SamplingParams(
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
    completed = ordinal_reasoning.validate_existing(existing, records, config_sha256)

    def score_records(batch: list[dict[str, Any]]) -> None:
        generation_prompts = []
        for row in batch:
            trajectory = extract_trajectory(
                str(row["prompt"]), binary_template.cache_prefix
            )
            user_prompt = render_ordinal_user_prompt(instruction, trajectory)
            generation_prompts.append(
                render_generation_prompt(
                    tokenizer,
                    user_prompt,
                    enable_thinking=bool(prompt_config["enable_thinking"]),
                    assistant_suffix=str(prompt_config["assistant_suffix"]),
                )
            )
        rationale_outputs = llm.generate(generation_prompts, reasoning_sampling)
        if len(rationale_outputs) != len(batch):
            raise RuntimeError("vLLM rationale count differs from request count")
        continuations = [
            ordinal_reasoning.reasoning_continuation(
                output, str(generation["stop"])
            )[0]
            for output in rationale_outputs
        ]
        margin_prompts = [
            prompt + continuation
            for prompt, continuation in zip(
                generation_prompts, continuations, strict=True
            )
        ]
        margin_outputs = llm.generate(margin_prompts, margin_sampling)
        if len(margin_outputs) != len(batch):
            raise RuntimeError("vLLM margin count differs from request count")
        for record, prompt, rationale, margin_prompt, margin_output in zip(
            batch,
            generation_prompts,
            rationale_outputs,
            margin_prompts,
            margin_outputs,
            strict=True,
        ):
            packed = reasoned_prediction_row(
                record,
                prompt,
                rationale,
                margin_prompt,
                margin_output,
                tokenizer,
                token_ids,
                str(generation["stop"]),
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
        raise RuntimeError("balanced reasoning canary failed")
    atomic_write_json(
        args.output / "canary_result.json",
        {
            "campaign_id": config["campaign_id"],
            "config_sha256": config_sha256,
            "purpose": "visible_reasoning_then_selected_binary_logprobs",
            **summarize_reasoning(canary_predictions),
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

    direct_path = Path(config["scoring"]["direct_baseline_predictions"])
    if sha256_file(direct_path) != config["scoring"]["direct_baseline_sha256"]:
        raise ValueError("direct baseline prediction checksum differs")
    direct_comparison = compare_direct_baseline(predictions, load_jsonl(direct_path))
    result = {
        "campaign_id": config["campaign_id"],
        "config_sha256": config_sha256,
        "model": model,
        "prompt": prompt_config,
        "generation": generation,
        "engine": engine,
        "runtime": {
            "vllm_version": vllm.__version__,
            "torch_version": torch.__version__,
            "gpu_name": torch.cuda.get_device_name(0),
            "elapsed_seconds_this_invocation": time.time() - started,
        },
        "score": "normalized literal 1 versus 0 logprob after visible reasoning",
        **summarize_reasoning(predictions),
        "direct_baseline": {
            "path": str(direct_path),
            "sha256": config["scoring"]["direct_baseline_sha256"],
            **direct_comparison,
        },
    }
    atomic_write_json(args.output / "result.json", result)
    macro = result["metrics"]["macro"]["macro"]
    delta = direct_comparison["delta_reasoned_minus_direct"]["macro"]
    print(
        f"complete rows={len(predictions)} "
        f"macro_auroc={macro['auroc']:.6f} "
        f"macro_pauroc_at_20={macro['pauroc_at_20']:.6f} "
        f"delta_direct_pauroc={delta['pauroc_at_20']:+.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
