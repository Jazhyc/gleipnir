#!/usr/bin/env python3
"""Benchmark GPT-OSS native low reasoning before direct OOD decision logits."""

from __future__ import annotations

import argparse
import hashlib
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.tool_trajectory_monitoring.benchmark_gpt_oss_ood import (
    balanced_canary_rows,
    batches,
    binary_logprobs,
    normalized_score,
    ordered_predictions,
    summarize,
    validate_inputs,
)
from experiments.tool_trajectory_monitoring.prompting import load_prompt_set
from experiments.tool_trajectory_monitoring.qwen_reasoning_core import (
    atomic_write_json,
    atomic_write_jsonl,
    load_json,
    load_jsonl,
    paired_condition_comparison,
)
from gleipnir.binary_evaluation import binary_token_ids
from gleipnir.qwen35_adapter_rebase import sha256_file

DEFAULT_CONFIG = Path(
    "experiments/tool_trajectory_monitoring/"
    "gpt_oss_low_reasoning_ood_benchmark.json"
)
DEFAULT_OUTPUT = Path(
    "results/tool_trajectory_monitoring/gpt_oss_120b_low_reasoning_ood"
)
HARMONY_ASSISTANT_START = "<|start|>assistant"
CHAT_TEMPLATE_DATE_EXPRESSION = 'strftime_now("%Y-%m-%d")'


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
    if config.get("model", {}).get("id") != "openai/gpt-oss-120b":
        raise ValueError("frozen model must be openai/gpt-oss-120b")
    prompt = config.get("prompt", {})
    expected_prompt = {
        "role": "teacher",
        "reasoning_effort_header": "low",
        "analysis_prefix": "<|channel|>analysis<|message|>",
        "analysis_stop": "<|end|>",
        "final_boundary": (
            "<|start|>assistant<|channel|>final<|message|>Prediction:"
        ),
        "current_date": "2026-08-30",
    }
    for key, expected in expected_prompt.items():
        if prompt.get(key) != expected:
            raise ValueError(f"frozen prompt field {key!r} differs")
    generation = config.get("generation", {})
    if (
        int(generation.get("max_tokens", 0)) != 16_384
        or float(generation.get("temperature", -1.0)) != 1.0
        or float(generation.get("top_p", -1.0)) != 1.0
    ):
        raise ValueError("frozen GPT-OSS generation settings differ")
    if config.get("scoring", {}).get("tokens") != ["0", "1"]:
        raise ValueError("frozen decision tokens must be literal 0 and 1")
    engine = config.get("engine", {})
    maximum = int(engine.get("audited_max_source_prompt_tokens", 0))
    generation_cap = int(generation["max_tokens"])
    if maximum + generation_cap >= int(engine.get("max_model_len", 0)):
        raise ValueError("engine context does not cover source plus reasoning cap")
    if int(engine.get("canary_rows_per_source_label", 0)) < 1:
        raise ValueError("canary rows per source-label must be positive")


def sha256_token_ids(token_ids: list[int]) -> str:
    payload = ",".join(str(token_id) for token_id in token_ids)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def render_reasoning_prompt(
    tokenizer: Any,
    user_prompt: str,
    *,
    reasoning_effort: str,
    current_date: str,
) -> str:
    chat_template = str(tokenizer.chat_template)
    if chat_template.count(CHAT_TEMPLATE_DATE_EXPRESSION) != 1:
        raise ValueError("GPT-OSS chat-template date expression differs")
    chat_template = chat_template.replace(
        CHAT_TEMPLATE_DATE_EXPRESSION,
        f'"{current_date}"',
    )
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        chat_template=chat_template,
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort=reasoning_effort,
    )
    if not rendered.endswith(HARMONY_ASSISTANT_START):
        raise ValueError(
            "GPT-OSS chat template no longer ends at the expected assistant start"
        )
    return rendered


def harmony_token_contract(
    tokenizer: Any,
    prompt_config: dict[str, Any],
) -> dict[str, Any]:
    def encoded(value: str) -> list[int]:
        return list(tokenizer.encode(value, add_special_tokens=False))

    analysis_prefix = encoded(str(prompt_config["analysis_prefix"]))
    final_boundary = encoded(str(prompt_config["final_boundary"]))
    analysis_stop = encoded(str(prompt_config["analysis_stop"]))
    if not analysis_prefix or not final_boundary or len(analysis_stop) != 1:
        raise ValueError("Harmony boundary tokenization differs from the contract")
    return {
        "analysis_prefix_ids": analysis_prefix,
        "analysis_stop_id": analysis_stop[0],
        "final_boundary_ids": final_boundary,
    }


def audit_source_prompts(
    tokenizer: Any,
    records: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, int]:
    reasoning_effort = str(config["prompt"]["reasoning_effort_header"])
    current_date = str(config["prompt"]["current_date"])
    lengths = []
    for record in records:
        prompt = render_reasoning_prompt(
            tokenizer,
            str(record["prompt"]),
            reasoning_effort=reasoning_effort,
            current_date=current_date,
        )
        lengths.append(len(tokenizer.encode(prompt, add_special_tokens=False)))
    observed_maximum = max(lengths)
    expected_maximum = int(config["engine"]["audited_max_source_prompt_tokens"])
    if observed_maximum != expected_maximum:
        raise ValueError(
            "source-prompt token maximum differs from frozen audit: "
            f"observed={observed_maximum} expected={expected_maximum}"
        )
    return {
        "rows": len(lengths),
        "minimum": min(lengths),
        "maximum": observed_maximum,
    }


def output_value(output: Any) -> Any:
    if not output.outputs:
        raise RuntimeError("vLLM returned no generated continuation")
    return output.outputs[0]


def pack_generation(
    record: dict[str, Any],
    prompt: str,
    prompt_token_ids: list[int],
    output: Any,
    contract: dict[str, Any],
    config_sha256: str,
) -> dict[str, Any]:
    value = output_value(output)
    generated = list(value.token_ids)
    stop_id = int(contract["analysis_stop_id"])
    stop_included = bool(generated and generated[-1] == stop_id)
    if stop_included:
        generated = generated[:-1]
    finish_reason = None if value.finish_reason is None else str(value.finish_reason)
    raw_stop_reason = value.stop_reason
    stop_reason = (
        None
        if raw_stop_reason is None
        else (
            int(raw_stop_reason)
            if isinstance(raw_stop_reason, int)
            else str(raw_stop_reason)
        )
    )
    valid_stop = stop_reason == stop_id or str(stop_reason) == str(stop_id)
    errors = []
    if finish_reason != "stop" or not valid_stop:
        errors.append("did_not_stop_at_harmony_end")
    if not generated:
        errors.append("empty_generation")
    metadata = record["metadata"]
    return {
        "id": str(record["id"]),
        "source": str(metadata["source_dataset"]),
        "label": int(metadata["ground_truth"]),
        "config_sha256": config_sha256,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_tokens": len(prompt_token_ids),
        "generation_token_ids": generated,
        "generation_token_ids_sha256": sha256_token_ids(generated),
        "generation_tokens": len(generated),
        "analysis_content_tokens": len(generated),
        "generation_text": str(value.text),
        "finish_reason": finish_reason,
        "stop_reason": stop_reason,
        "stop_token_included_by_backend": stop_included,
        "format_errors": errors,
        "valid": not errors,
    }


def validate_existing_rows(
    rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    config_sha256: str,
    *,
    kind: str,
) -> dict[str, dict[str, Any]]:
    allowed = {str(record["id"]): record for record in records}
    completed: dict[str, dict[str, Any]] = {}
    for row in rows:
        record_id = str(row.get("id"))
        if record_id not in allowed:
            raise ValueError(f"existing {kind} contains unknown id={record_id!r}")
        if row.get("config_sha256") != config_sha256:
            raise ValueError(f"existing {kind} config drift for id={record_id!r}")
        metadata = allowed[record_id]["metadata"]
        if row.get("source") != metadata["source_dataset"]:
            raise ValueError(f"existing {kind} source drift for id={record_id!r}")
        if int(row.get("label")) != int(metadata["ground_truth"]):
            raise ValueError(f"existing {kind} label drift for id={record_id!r}")
        if record_id in completed:
            raise ValueError(f"existing {kind} contains duplicate id={record_id!r}")
        if kind == "generation" and not bool(row.get("valid")):
            raise ValueError(f"existing generation is invalid for id={record_id!r}")
        if kind == "generation":
            token_ids = [int(value) for value in row.get("generation_token_ids", [])]
            if not token_ids:
                raise ValueError(
                    f"existing generation has no token IDs for id={record_id!r}"
                )
            if sha256_token_ids(token_ids) != row.get("generation_token_ids_sha256"):
                raise ValueError(
                    f"existing generation token hash differs for id={record_id!r}"
                )
        if kind == "prediction" and not math.isfinite(float(row.get("score"))):
            raise ValueError(
                f"existing prediction score is invalid for id={record_id!r}"
            )
        completed[record_id] = row
    return completed


def build_margin_token_ids(
    prompt_token_ids: list[int],
    generation_token_ids: list[int],
    contract: dict[str, Any],
) -> list[int]:
    return [
        *prompt_token_ids,
        *contract["analysis_prefix_ids"],
        *generation_token_ids,
        int(contract["analysis_stop_id"]),
        *contract["final_boundary_ids"],
    ]


def generate_records(
    llm: Any,
    tokenizer: Any,
    sampling: Any,
    target_records: list[dict[str, Any]],
    all_records: list[dict[str, Any]],
    completed: dict[str, dict[str, Any]],
    generations_path: Path,
    config: dict[str, Any],
    config_sha256: str,
    contract: dict[str, Any],
) -> None:
    pending = [row for row in target_records if str(row["id"]) not in completed]
    batch_size = int(config["engine"]["batch_rows"])
    reasoning_effort = str(config["prompt"]["reasoning_effort_header"])
    current_date = str(config["prompt"]["current_date"])
    for batch_index, batch in enumerate(batches(pending, batch_size), start=1):
        prompts = [
            render_reasoning_prompt(
                tokenizer,
                str(record["prompt"]),
                reasoning_effort=reasoning_effort,
                current_date=current_date,
            )
            for record in batch
        ]
        prompt_ids = [
            list(tokenizer.encode(prompt, add_special_tokens=False))
            for prompt in prompts
        ]
        reasoning_prompt_ids = [
            [*token_ids, *contract["analysis_prefix_ids"]]
            for token_ids in prompt_ids
        ]
        outputs = llm.generate(
            [{"prompt_token_ids": token_ids} for token_ids in reasoning_prompt_ids],
            sampling,
        )
        if len(outputs) != len(batch):
            raise RuntimeError("vLLM generation count differs from request count")
        packed = [
            pack_generation(
                record,
                prompt,
                token_ids,
                output,
                contract,
                config_sha256,
            )
            for record, prompt, token_ids, output in zip(
                batch, prompts, prompt_ids, outputs, strict=True
            )
        ]
        for row in packed:
            completed[row["id"]] = row
        atomic_write_jsonl(
            generations_path,
            ordered_predictions(all_records, completed),
        )
        invalid = [row for row in packed if not row["valid"]]
        if invalid:
            details = {row["id"]: row["format_errors"] for row in invalid}
            raise RuntimeError(f"invalid Harmony reasoning generations: {details}")
        print(
            f"generation batch={batch_index} "
            f"complete={len(completed)}/{len(all_records)}",
            flush=True,
        )


def score_records(
    llm: Any,
    tokenizer: Any,
    sampling: Any,
    target_records: list[dict[str, Any]],
    all_records: list[dict[str, Any]],
    generations: dict[str, dict[str, Any]],
    completed: dict[str, dict[str, Any]],
    predictions_path: Path,
    config: dict[str, Any],
    config_sha256: str,
    contract: dict[str, Any],
    decision_token_ids: list[int],
) -> None:
    pending = [row for row in target_records if str(row["id"]) not in completed]
    batch_size = int(config["engine"]["batch_rows"])
    reasoning_effort = str(config["prompt"]["reasoning_effort_header"])
    current_date = str(config["prompt"]["current_date"])
    max_model_len = int(config["engine"]["max_model_len"])
    for batch_index, batch in enumerate(batches(pending, batch_size), start=1):
        prompts = [
            render_reasoning_prompt(
                tokenizer,
                str(record["prompt"]),
                reasoning_effort=reasoning_effort,
                current_date=current_date,
            )
            for record in batch
        ]
        source_prompt_ids = [
            list(tokenizer.encode(prompt, add_special_tokens=False))
            for prompt in prompts
        ]
        for record, prompt, prompt_ids in zip(
            batch, prompts, source_prompt_ids, strict=True
        ):
            generated = generations[str(record["id"])]
            expected_prompt_sha256 = hashlib.sha256(
                prompt.encode("utf-8")
            ).hexdigest()
            if generated["prompt_sha256"] != expected_prompt_sha256:
                raise ValueError(
                    f"reasoning prompt hash differs for id={record['id']!r}"
                )
            if int(generated["prompt_tokens"]) != len(prompt_ids):
                raise ValueError(
                    f"reasoning prompt length differs for id={record['id']!r}"
                )
        margin_ids = [
            build_margin_token_ids(
                prompt_ids,
                list(generations[str(record["id"])]["generation_token_ids"]),
                contract,
            )
            for record, prompt_ids in zip(batch, source_prompt_ids, strict=True)
        ]
        too_long = [
            (str(record["id"]), len(token_ids))
            for record, token_ids in zip(batch, margin_ids, strict=True)
            if len(token_ids) >= max_model_len
        ]
        if too_long:
            raise RuntimeError(
                f"reasoning margin prompts reach max_model_len: {too_long}"
            )
        outputs = llm.generate(
            [{"prompt_token_ids": token_ids} for token_ids in margin_ids],
            sampling,
        )
        if len(outputs) != len(batch):
            raise RuntimeError("vLLM margin count differs from request count")
        for record, prompt, prompt_ids, conditioned_ids, output in zip(
            batch, prompts, source_prompt_ids, margin_ids, outputs, strict=True
        ):
            record_id = str(record["id"])
            generated = generations[record_id]
            logprob_0, logprob_1 = binary_logprobs(output, decision_token_ids)
            score = normalized_score(logprob_0, logprob_1)
            decision = list(output_value(output).token_ids)
            if len(decision) != 1 or decision[0] not in decision_token_ids:
                raise RuntimeError(f"invalid constrained decision for id={record_id!r}")
            conceptual_output_tokens = (
                len(contract["analysis_prefix_ids"])
                + int(generated["generation_tokens"])
                + 1
                + len(contract["final_boundary_ids"])
                + 1
            )
            metadata = record["metadata"]
            row = {
                "id": record_id,
                "source": str(metadata["source_dataset"]),
                "label": int(metadata["ground_truth"]),
                "condition": "native_low_analysis_then_direct_final_logits",
                "config_sha256": config_sha256,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "prompt_tokens": len(prompt_ids),
                "generation_token_ids_sha256": generated[
                    "generation_token_ids_sha256"
                ],
                "generation_tokens": int(generated["generation_tokens"]),
                "analysis_content_tokens": int(
                    generated["analysis_content_tokens"]
                ),
                "conceptual_output_tokens": conceptual_output_tokens,
                "second_pass_prompt_tokens": len(conditioned_ids),
                "margin_prompt_token_ids_sha256": sha256_token_ids(conditioned_ids),
                "score": score,
                "prediction": int(decision[0] == decision_token_ids[1]),
                "logprob_0": logprob_0,
                "logprob_1": logprob_1,
                "logit_margin_1_minus_0": logprob_1 - logprob_0,
                "token_id_0": decision_token_ids[0],
                "token_id_1": decision_token_ids[1],
            }
            completed[record_id] = row
        atomic_write_jsonl(
            predictions_path,
            ordered_predictions(all_records, completed),
        )
        print(
            f"scoring batch={batch_index} "
            f"complete={len(completed)}/{len(all_records)}",
            flush=True,
        )


def summarize_reasoning(
    predictions: list[dict[str, Any]],
    generations: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = summarize(predictions)
    prediction_frame = pd.DataFrame(predictions)
    generation_frame = pd.DataFrame(generations)
    scores = prediction_frame["score"].to_numpy(dtype=float)
    clipped = np.clip(scores, 1e-15, 1.0 - 1e-15)
    entropy = -(clipped * np.log2(clipped) + (1.0 - clipped) * np.log2(1.0 - clipped))
    summary.update(
        {
            "generation": {
                "rows": len(generation_frame),
                "tokens_total": int(generation_frame["generation_tokens"].sum()),
                "tokens_mean": float(generation_frame["generation_tokens"].mean()),
                "tokens_max": int(generation_frame["generation_tokens"].max()),
                "analysis_content_tokens_total": int(
                    generation_frame["analysis_content_tokens"].sum()
                ),
                "format_failures": int((~generation_frame["valid"]).sum()),
                "truncated": int(
                    (generation_frame["finish_reason"] == "length").sum()
                ),
            },
            "conceptual_token_accounting": {
                "input_tokens_total": int(prediction_frame["prompt_tokens"].sum()),
                "input_tokens_mean": float(prediction_frame["prompt_tokens"].mean()),
                "output_tokens_total": int(
                    prediction_frame["conceptual_output_tokens"].sum()
                ),
                "output_tokens_mean": float(
                    prediction_frame["conceptual_output_tokens"].mean()
                ),
                "note": (
                    "One-pass hosted equivalent: sampled analysis plus Harmony "
                    "transition and one decision token; excludes the local second "
                    "forward pass from token-billed cost."
                ),
            },
            "uncertainty": {
                "mean_binary_entropy_bits": float(np.mean(entropy)),
                "mean_confidence": float(
                    np.mean(np.maximum(scores, 1.0 - scores))
                ),
            },
        }
    )
    return summary


def validate_baseline(
    config: dict[str, Any],
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline = config["baseline"]
    predictions_path = Path(baseline["predictions"])
    result_path = Path(baseline["result"])
    if sha256_file(predictions_path) != baseline["predictions_sha256"]:
        raise ValueError("immediate-logit prediction baseline hash differs")
    if sha256_file(result_path) != baseline["result_sha256"]:
        raise ValueError("immediate-logit result baseline hash differs")
    rows = load_jsonl(predictions_path)
    expected_ids = {str(record["id"]) for record in records}
    if len(rows) != len(records) or {str(row["id"]) for row in rows} != expected_ids:
        raise ValueError("immediate-logit baseline row IDs differ")
    return rows


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    validate_config(config)
    records = validate_inputs(config)
    baseline_rows = validate_baseline(config, records)
    config_sha256 = sha256_file(args.config)
    prompt_set = load_prompt_set()
    if prompt_set.teacher.template_sha256 != config["prompt"]["template_sha256"]:
        raise ValueError("working-tree teacher prompt differs from frozen config")

    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model = config["model"]
    engine = config["engine"]
    generation_config = config["generation"]
    tokenizer = AutoTokenizer.from_pretrained(
        model["id"], revision=model["revision"]
    )
    source_prompt_audit = audit_source_prompts(tokenizer, records, config)
    decision_token_ids = binary_token_ids(tokenizer)
    contract = harmony_token_contract(tokenizer, config["prompt"])
    generation_sampling = SamplingParams(
        max_tokens=int(generation_config["max_tokens"]),
        temperature=float(generation_config["temperature"]),
        top_p=float(generation_config["top_p"]),
        seed=int(generation_config["seed"]),
        stop_token_ids=[int(contract["analysis_stop_id"])],
    )
    margin_sampling = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=len(decision_token_ids),
        logprob_token_ids=decision_token_ids,
        allowed_token_ids=decision_token_ids,
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
        seed=int(generation_config["seed"]),
    )

    generations_path = args.output / "generations.jsonl"
    predictions_path = args.output / "predictions.jsonl"
    existing_generations = (
        []
        if args.force or not generations_path.is_file()
        else load_jsonl(generations_path)
    )
    existing_predictions = (
        []
        if args.force or not predictions_path.is_file()
        else load_jsonl(predictions_path)
    )
    generations = validate_existing_rows(
        existing_generations,
        records,
        config_sha256,
        kind="generation",
    )
    predictions = validate_existing_rows(
        existing_predictions,
        records,
        config_sha256,
        kind="prediction",
    )
    canary = balanced_canary_rows(
        records,
        list(config["scope"]["sources"]),
        rows_per_source_label=int(engine["canary_rows_per_source_label"]),
    )
    started = time.time()
    generate_records(
        llm,
        tokenizer,
        generation_sampling,
        canary,
        records,
        generations,
        generations_path,
        config,
        config_sha256,
        contract,
    )
    score_records(
        llm,
        tokenizer,
        margin_sampling,
        canary,
        records,
        generations,
        predictions,
        predictions_path,
        config,
        config_sha256,
        contract,
        decision_token_ids,
    )
    canary_predictions = [predictions[str(row["id"])] for row in canary]
    canary_generations = [generations[str(row["id"])] for row in canary]
    if not all(math.isfinite(float(row["score"])) for row in canary_predictions):
        raise RuntimeError("balanced reasoning canary returned a non-finite score")
    atomic_write_json(
        args.output / "canary_result.json",
        {
            "campaign_id": config["campaign_id"],
            "config_sha256": config_sha256,
            "purpose": "native_low_reasoning_and_selected_logprob_canary",
            **summarize_reasoning(canary_predictions, canary_generations),
        },
    )
    print(f"canary complete rows={len(canary_predictions)}", flush=True)
    if args.stop_after_canary:
        return

    generate_records(
        llm,
        tokenizer,
        generation_sampling,
        records,
        records,
        generations,
        generations_path,
        config,
        config_sha256,
        contract,
    )
    score_records(
        llm,
        tokenizer,
        margin_sampling,
        records,
        records,
        generations,
        predictions,
        predictions_path,
        config,
        config_sha256,
        contract,
        decision_token_ids,
    )
    generation_rows = ordered_predictions(records, generations)
    prediction_rows = ordered_predictions(records, predictions)
    if len(generation_rows) != len(records) or len(prediction_rows) != len(records):
        raise RuntimeError("full GPT-OSS reasoning benchmark is incomplete")
    result = {
        "campaign_id": config["campaign_id"],
        "config_sha256": config_sha256,
        "model": model,
        "prompt": config["prompt"],
        "generation_settings": generation_config,
        "engine": engine,
        "source_prompt_audit": source_prompt_audit,
        "runtime": {
            "vllm_version": vllm.__version__,
            "torch_version": torch.__version__,
            "gpu_name": torch.cuda.get_device_name(0),
            "elapsed_seconds_this_invocation": time.time() - started,
        },
        "score": (
            "normalized direct final-channel probability for literal 1 versus 0 "
            "after sampled native low-effort analysis"
        ),
        **summarize_reasoning(prediction_rows, generation_rows),
        "comparison_to_immediate_logits": paired_condition_comparison(
            baseline_rows, prediction_rows
        ),
    }
    atomic_write_json(args.output / "result.json", result)
    macro = result["metrics"]["macro"]["macro"]
    pooled = result["metrics"]["pooled"]
    print(
        f"complete rows={len(prediction_rows)} "
        f"macro_auroc={macro['auroc']:.6f} "
        f"macro_pauroc_at_20={macro['pauroc_at_20']:.6f} "
        f"pooled_auroc={pooled['auroc']:.6f} "
        f"pooled_pauroc_at_20={pooled['pauroc_at_20']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
