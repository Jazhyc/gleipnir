#!/usr/bin/env python3
"""Benchmark visible reasoning followed by a 0--10 Qwen OOD score."""

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

from experiments.tool_trajectory_monitoring.benchmark_gpt_oss_ood import (
    balanced_canary_rows,
    batches,
    ordered_predictions,
    validate_inputs,
)
from experiments.tool_trajectory_monitoring.benchmark_qwen_ood import (
    QWEN_NON_THINKING_ASSISTANT_SUFFIX,
)
from experiments.tool_trajectory_monitoring.benchmark_qwen_ordinal_ood import (
    compare_binary_baseline,
    extract_trajectory,
    render_generation_prompt,
    render_ordinal_user_prompt,
    summarize,
    validate_choice_tokenization,
)
from experiments.tool_trajectory_monitoring.prompting import load_prompt_set
from experiments.tool_trajectory_monitoring.qwen_reasoning_core import (
    atomic_write_json,
    atomic_write_jsonl,
    load_json,
    load_jsonl,
)
from gleipnir.qwen35_adapter_rebase import sha256_file

DEFAULT_CONFIG = Path(
    "experiments/tool_trajectory_monitoring/"
    "qwen_ordinal_reasoning_ood_benchmark.json"
)
DEFAULT_OUTPUT = Path(
    "results/tool_trajectory_monitoring/"
    "qwen35_9b_teacher_ordinal_reasoning_ood_v3_final"
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
        raise ValueError("ordinal reasoning instruction file is missing")
    if sha256_file(instruction_path) != prompt.get("instruction_sha256"):
        raise ValueError("ordinal reasoning instruction hash differs")

    scoring = config.get("scoring", {})
    if scoring.get("choices") != [str(score) for score in range(11)]:
        raise ValueError("frozen ordinal choices must be literal integers 0--10")
    generation = config.get("generation", {})
    if generation.get("stop") != "\nScore:":
        raise ValueError("visible reasoning must stop at the terminal score boundary")
    if generation.get("include_stop_str_in_output") is not True:
        raise ValueError("score boundary must remain in the visible continuation")
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
    if audited_max + max_rationale + 2 > max_model_len:
        raise ValueError("max_model_len does not cover prompt, rationale, and score")
    if int(engine.get("canary_rows_per_source_label", 0)) < 1:
        raise ValueError("canary rows per source-label must be positive")
    if engine.get("language_model_only") is not True:
        raise ValueError("Qwen benchmark must use the text-only language model")


def validate_existing(
    existing: list[dict[str, Any]],
    records: list[dict[str, Any]],
    config_sha256: str,
) -> dict[str, dict[str, Any]]:
    allowed = {str(row["id"]): row for row in records}
    completed: dict[str, dict[str, Any]] = {}
    for row in existing:
        record_id = str(row.get("id"))
        if record_id not in allowed:
            raise ValueError(f"existing predictions contain unknown id={record_id!r}")
        if row.get("config_sha256") != config_sha256:
            raise ValueError(f"existing prediction config drift for id={record_id!r}")
        metadata = allowed[record_id]["metadata"]
        if row.get("source") != metadata["source_dataset"]:
            raise ValueError(f"existing prediction source drift for id={record_id!r}")
        if int(row.get("label")) != int(metadata["ground_truth"]):
            raise ValueError(f"existing prediction label drift for id={record_id!r}")
        if record_id in completed:
            raise ValueError(f"existing predictions contain duplicate id={record_id!r}")
        completed[record_id] = row
    return completed


def reasoning_continuation(output: Any, stop: str) -> tuple[str, list[int], str]:
    if not output.outputs:
        raise RuntimeError("vLLM returned no visible reasoning generation")
    generated = output.outputs[0]
    text = str(generated.text)
    token_ids = list(generated.token_ids)
    if not token_ids:
        raise RuntimeError("visible reasoning returned no token IDs")
    if text.endswith(stop) and text.count(stop) == 1:
        rationale = text[: -len(stop)].strip()
        recovery = "none"
    elif str(generated.finish_reason) == "length":
        rationale = text.strip()
        text = text.rstrip() + stop
        recovery = "length_cap"
    elif str(generated.finish_reason) == "stop":
        rationale = text.strip()
        text = text.rstrip() + stop
        recovery = "missing_terminal_boundary"
    else:
        text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        raise RuntimeError(
            "visible reasoning cannot be continued at the score boundary: "
            f"finish_reason={generated.finish_reason!r} "
            f"stop_reason={generated.stop_reason!r} "
            f"tokens={len(generated.token_ids)} text_sha256={text_sha256}"
        )
    if not rationale:
        raise RuntimeError("visible reasoning produced no analysis before the score")
    if recovery == "none" and str(generated.finish_reason) != "stop":
        raise RuntimeError(
            f"visible reasoning finish reason is {generated.finish_reason!r}"
        )
    return text, token_ids, recovery


def reasoned_prediction_row(
    record: dict[str, Any],
    generation_prompt: str,
    rationale_output: Any,
    score_prompt: str,
    score_output: Any,
    tokenizer: Any,
    choice_token_ids: dict[str, list[int]],
    eos_token_id: int,
    stop: str,
    config_sha256: str,
    max_model_len: int,
) -> dict[str, Any]:
    continuation, rationale_token_ids, boundary_recovery = reasoning_continuation(
        rationale_output, stop
    )
    if score_prompt != generation_prompt + continuation:
        raise RuntimeError("score prompt is not the exact reasoning continuation")
    if not score_output.outputs:
        raise RuntimeError("vLLM returned no ordinal score")
    generated_score = score_output.outputs[0]
    score_text = str(generated_score.text)
    if score_text not in choice_token_ids:
        raise RuntimeError(f"invalid constrained ordinal score {score_text!r}")
    score_output_token_ids = list(generated_score.token_ids)
    score_token_ids = choice_token_ids[score_text]
    termination_token_ids = score_output_token_ids[len(score_token_ids) :]
    if (
        score_output_token_ids[: len(score_token_ids)] != score_token_ids
        or termination_token_ids not in ([], [eos_token_id])
    ):
        raise RuntimeError(
            f"ordinal score tokenization drift for {score_text!r}: "
            f"{score_output_token_ids!r}"
        )

    prompt_token_ids = list(rationale_output.prompt_token_ids or [])
    score_prompt_token_ids = list(score_output.prompt_token_ids or [])
    if not prompt_token_ids or not score_prompt_token_ids:
        raise RuntimeError("vLLM omitted prompt token IDs")
    if len(score_prompt_token_ids) + len(score_output_token_ids) > max_model_len:
        raise RuntimeError("reasoning plus score reaches max_model_len")
    reasoning_context_tokens = len(score_prompt_token_ids) - len(prompt_token_ids)
    synthetic_boundary_tokens = (
        len(tokenizer.encode(stop, add_special_tokens=False))
        if boundary_recovery != "none"
        else 0
    )
    visible_completion_tokens = len(
        tokenizer.encode(continuation + score_text, add_special_tokens=False)
    )
    logical_output_tokens = visible_completion_tokens + len(termination_token_ids)

    ordinal_score = int(score_text)
    metadata = record["metadata"]
    rationale = continuation[: -len(stop)].strip()
    return {
        "id": str(record["id"]),
        "source": str(metadata["source_dataset"]),
        "label": int(metadata["ground_truth"]),
        "score": ordinal_score / 10.0,
        "ordinal_score": ordinal_score,
        "prediction": int(ordinal_score >= 5),
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
        "generated_text": score_text,
        "generated_token_ids": score_output_token_ids,
        "score_token_ids": score_token_ids,
        "termination_token_ids": termination_token_ids,
        "score_tokens": len(score_token_ids),
        "output_tokens": logical_output_tokens,
        "served_output_tokens": len(rationale_token_ids)
        + len(score_output_token_ids),
        "completion_retokenization_delta": logical_output_tokens
        - len(rationale_token_ids)
        - synthetic_boundary_tokens
        - len(score_output_token_ids),
        "prompt_tokens": len(prompt_token_ids),
        "score_context_tokens": len(score_prompt_token_ids),
        "source_prompt_sha256": metadata["rendered_prompt_sha256"],
        "generation_prompt_sha256": hashlib.sha256(
            generation_prompt.encode("utf-8")
        ).hexdigest(),
        "score_prompt_sha256": hashlib.sha256(
            score_prompt.encode("utf-8")
        ).hexdigest(),
        "config_sha256": config_sha256,
    }


def summarize_reasoning(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize(rows)
    frame = pd.DataFrame(rows)
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
    summary["score_context_tokens"] = {
        "total": int(frame["score_context_tokens"].sum()),
        "min": int(frame["score_context_tokens"].min()),
        "max": int(frame["score_context_tokens"].max()),
        "mean": float(frame["score_context_tokens"].mean()),
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


def compare_immediate_ordinal(
    reasoned_rows: list[dict[str, Any]], immediate_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    reasoned_by_id = {str(row["id"]): row for row in reasoned_rows}
    immediate_by_id = {str(row["id"]): row for row in immediate_rows}
    if len(reasoned_by_id) != len(reasoned_rows) or len(immediate_by_id) != len(
        immediate_rows
    ):
        raise ValueError("paired comparison contains duplicate IDs")
    if reasoned_by_id.keys() != immediate_by_id.keys():
        raise ValueError("reasoned and immediate ordinal IDs differ")
    for record_id, reasoned in reasoned_by_id.items():
        immediate = immediate_by_id[record_id]
        if reasoned["source"] != immediate["source"] or int(
            reasoned["label"]
        ) != int(immediate["label"]):
            raise ValueError(f"paired baseline identity drift for id={record_id!r}")

    reasoned_summary = summarize(reasoned_rows)
    immediate_summary = summarize(immediate_rows)
    reasoned_scores = np.asarray(
        [float(reasoned_by_id[record_id]["score"]) for record_id in reasoned_by_id]
    )
    immediate_scores = np.asarray(
        [float(immediate_by_id[record_id]["score"]) for record_id in reasoned_by_id]
    )
    correlation = (
        float(np.corrcoef(reasoned_scores, immediate_scores)[0, 1])
        if np.std(reasoned_scores) > 0 and np.std(immediate_scores) > 0
        else None
    )
    return {
        "immediate_metrics": immediate_summary["metrics"],
        "delta_reasoned_minus_immediate": _metric_delta(
            reasoned_summary["metrics"], immediate_summary["metrics"]
        ),
        "score_agreement": float(np.mean(reasoned_scores == immediate_scores)),
        "mean_absolute_score_delta": float(
            np.mean(np.abs(reasoned_scores - immediate_scores))
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
    from vllm.sampling_params import StructuredOutputsParams

    model = config["model"]
    engine = config["engine"]
    prompt_config = config["prompt"]
    generation = config["generation"]
    choices = list(config["scoring"]["choices"])
    tokenizer = AutoTokenizer.from_pretrained(
        model["id"], revision=model["revision"]
    )
    choice_token_ids = validate_choice_tokenization(tokenizer, choices)
    reasoning_sampling = SamplingParams(
        max_tokens=int(generation["max_tokens"]),
        temperature=float(generation["temperature"]),
        stop=[str(generation["stop"])],
        include_stop_str_in_output=bool(generation["include_stop_str_in_output"]),
    )
    score_sampling = SamplingParams(
        max_tokens=2,
        temperature=0.0,
        structured_outputs=StructuredOutputsParams(choice=choices),
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
            reasoning_continuation(output, str(generation["stop"]))[0]
            for output in rationale_outputs
        ]
        score_prompts = [
            prompt + continuation
            for prompt, continuation in zip(
                generation_prompts, continuations, strict=True
            )
        ]
        score_outputs = llm.generate(score_prompts, score_sampling)
        if len(score_outputs) != len(batch):
            raise RuntimeError("vLLM score count differs from request count")
        for record, prompt, rationale, score_prompt, score_output in zip(
            batch,
            generation_prompts,
            rationale_outputs,
            score_prompts,
            score_outputs,
            strict=True,
        ):
            packed = reasoned_prediction_row(
                record,
                prompt,
                rationale,
                score_prompt,
                score_output,
                tokenizer,
                choice_token_ids,
                int(tokenizer.eos_token_id),
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
            "purpose": "visible_reasoning_and_terminal_ordinal_interface_only",
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

    binary_path = Path(config["scoring"]["binary_baseline_predictions"])
    immediate_path = Path(config["scoring"]["immediate_ordinal_predictions"])
    if sha256_file(binary_path) != config["scoring"]["binary_baseline_sha256"]:
        raise ValueError("binary baseline prediction checksum differs")
    if sha256_file(immediate_path) != config["scoring"]["immediate_ordinal_sha256"]:
        raise ValueError("immediate ordinal prediction checksum differs")
    binary_comparison = compare_binary_baseline(
        predictions, load_jsonl(binary_path)
    )
    immediate_comparison = compare_immediate_ordinal(
        predictions, load_jsonl(immediate_path)
    )
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
        "score": "greedy visible reasoning followed by paper-rubric integer / 10",
        **summarize_reasoning(predictions),
        "binary_baseline": {
            "path": str(binary_path),
            "sha256": config["scoring"]["binary_baseline_sha256"],
            **binary_comparison,
        },
        "immediate_ordinal_baseline": {
            "path": str(immediate_path),
            "sha256": config["scoring"]["immediate_ordinal_sha256"],
            **immediate_comparison,
        },
    }
    atomic_write_json(args.output / "result.json", result)
    macro = result["metrics"]["macro"]["macro"]
    binary_delta = binary_comparison["delta_ordinal_minus_binary"]["macro"]
    immediate_delta = immediate_comparison[
        "delta_reasoned_minus_immediate"
    ]["macro"]
    print(
        f"complete rows={len(predictions)} "
        f"macro_auroc={macro['auroc']:.6f} "
        f"macro_pauroc_at_20={macro['pauroc_at_20']:.6f} "
        f"delta_binary_pauroc={binary_delta['pauroc_at_20']:+.6f} "
        f"delta_immediate_pauroc={immediate_delta['pauroc_at_20']:+.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
