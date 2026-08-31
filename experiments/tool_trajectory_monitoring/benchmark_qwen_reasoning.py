#!/usr/bin/env python3
"""Run direct, visible-reasoning, and thinking-mode Qwen3.5 logit arms."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from experiments.tool_trajectory_monitoring.prompting import load_prompt_set
from experiments.tool_trajectory_monitoring.qwen_reasoning_core import (
    atomic_write_json,
    atomic_write_jsonl,
    load_json,
    load_jsonl,
    paired_condition_comparison,
    prefix_before_terminal_prediction,
    reasoning_prompt,
    sha256_text,
    summarize_predictions,
)
from gleipnir.binary_evaluation import binary_token_ids, score_from_output
from gleipnir.qwen35_adapter_rebase import sha256_file

DEFAULT_CONFIG = Path(
    "experiments/tool_trajectory_monitoring/qwen_reasoning_benchmark.json"
)
DEFAULT_INPUT = Path("data/tool_trajectory_monitoring/qwen_reasoning/records.jsonl")
DEFAULT_OUTPUT = Path("results/tool_trajectory_monitoring/qwen_reasoning")
REASONING_INSTRUCTION = (
    Path(__file__).resolve().parent / "prompts" / "reasoning_evaluation.txt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--condition",
        action="append",
        choices=("direct", "visible_reasoning", "thinking_reasoning"),
        help="Run one or more conditions; default is all frozen conditions.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def validate_config(config: dict[str, Any]) -> None:
    conditions = config.get("conditions")
    names = [condition.get("name") for condition in conditions or []]
    expected = ["direct", "visible_reasoning", "thinking_reasoning"]
    if names != expected:
        raise ValueError(f"frozen condition order must be {expected}, got {names}")
    visible, thinking = conditions[1], conditions[2]
    for key in ("prompt", "generation"):
        if visible.get(key) != thinking.get(key):
            raise ValueError(f"reasoning conditions differ on {key}")
    if visible.get("enable_thinking") is not False:
        raise ValueError("visible_reasoning must disable native thinking")
    if thinking.get("enable_thinking") is not True:
        raise ValueError("thinking_reasoning must enable native thinking")
    if int(config["generation"]["max_tokens"]) != 16_384:
        raise ValueError("frozen generation cap must be 16,384 tokens")


def rendered_prompt(
    tokenizer: Any,
    trajectory: str,
    condition: dict[str, Any],
    reasoning_instruction: str,
) -> str:
    template = load_prompt_set().student
    if condition["name"] == "direct":
        user_prompt = template.render(trajectory)
    else:
        user_prompt = reasoning_prompt(
            template,
            trajectory,
            reasoning_instruction,
        )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=bool(condition["enable_thinking"]),
    )


def output_text(output: Any) -> str:
    if not output.outputs:
        return ""
    return str(output.outputs[0].text)


def output_token_count(output: Any) -> int:
    if not output.outputs:
        return 0
    return len(output.outputs[0].token_ids)


def output_finish_reason(output: Any) -> str | None:
    if not output.outputs:
        return None
    value = output.outputs[0].finish_reason
    return None if value is None else str(value)


def output_stop_reason(output: Any) -> str | int | None:
    if not output.outputs:
        return None
    value = output.outputs[0].stop_reason
    if value is None or isinstance(value, str | int):
        return value
    return str(value)


def batches(rows: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    if size < 1:
        raise ValueError("batch size must be positive")
    return [rows[start : start + size] for start in range(0, len(rows), size)]


def ordered_rows(
    records: list[dict[str, Any]],
    completed: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        completed[str(record["id"])]
        for record in records
        if record["id"] in completed
    ]


def validate_existing(
    existing: list[dict[str, Any]],
    records: list[dict[str, Any]],
    condition_name: str,
    config_sha256: str,
) -> dict[str, dict[str, Any]]:
    allowed = {str(record["id"]): record for record in records}
    completed: dict[str, dict[str, Any]] = {}
    for row in existing:
        record_id = str(row.get("id"))
        if record_id not in allowed:
            raise ValueError(f"existing result contains unknown id={record_id!r}")
        if row.get("condition") != condition_name:
            raise ValueError(f"existing result condition drift for id={record_id!r}")
        if row.get("config_sha256") != config_sha256:
            raise ValueError(f"existing result config drift for id={record_id!r}")
        if int(row.get("label")) != int(allowed[record_id]["label"]):
            raise ValueError(f"existing result label drift for id={record_id!r}")
        if record_id in completed:
            raise ValueError(f"existing result contains duplicate id={record_id!r}")
        completed[record_id] = row
    return completed


def run_condition(
    llm: Any,
    tokenizer: Any,
    generation_sampling: Any,
    margin_sampling: Any,
    records: list[dict[str, Any]],
    condition: dict[str, Any],
    reasoning_instruction: str,
    output_dir: Path,
    config: dict[str, Any],
    config_sha256: str,
    token_ids: list[int],
    *,
    force: bool,
) -> list[dict[str, Any]]:
    condition_name = str(condition["name"])
    predictions_path = output_dir / condition_name / "predictions.jsonl"
    existing = (
        []
        if force or not predictions_path.is_file()
        else load_jsonl(predictions_path)
    )
    completed = validate_existing(
        existing,
        records,
        condition_name,
        config_sha256,
    )
    pending = [record for record in records if str(record["id"]) not in completed]
    batch_size = int(config["engine"]["batch_rows"])
    started = time.time()
    for batch_index, batch in enumerate(batches(pending, batch_size), start=1):
        prompts = [
            rendered_prompt(
                tokenizer,
                str(record["trajectory"]),
                condition,
                reasoning_instruction,
            )
            for record in batch
        ]
        if condition["generation"]:
            generated_outputs = llm.generate(prompts, generation_sampling)
            generations = [output_text(output) for output in generated_outputs]
            parsed = [
                prefix_before_terminal_prediction(generation)
                for generation in generations
            ]
            margin_prompts = [
                prompt + prefix
                for prompt, (prefix, _) in zip(prompts, parsed, strict=True)
            ]
            margin_outputs = llm.generate(margin_prompts, margin_sampling)
        else:
            generations = [""] * len(batch)
            generated_outputs = [None] * len(batch)
            parsed = [("Prediction:", None)] * len(batch)
            margin_prompts = [prompt + "Prediction:" for prompt in prompts]
            margin_outputs = llm.generate(margin_prompts, margin_sampling)
        packed = zip(
            batch,
            prompts,
            generations,
            generated_outputs,
            parsed,
            margin_prompts,
            margin_outputs,
            strict=True,
        )
        for (
            record,
            prompt,
            generation,
            generated,
            parsed_value,
            margin_prompt,
            margin,
        ) in packed:
            _, generated_prediction = parsed_value
            stop_reason = (
                None if generated is None else output_stop_reason(generated)
            )
            row = {
                "id": str(record["id"]),
                "source": str(record["source"]),
                "label": int(record["label"]),
                "condition": condition_name,
                "enable_thinking": bool(condition["enable_thinking"]),
                "config_sha256": config_sha256,
                "prompt_sha256": sha256_text(prompt),
                "prompt_tokens": len(
                    tokenizer.encode(prompt, add_special_tokens=False)
                ),
                "generation": generation,
                "generation_sha256": sha256_text(generation),
                "generation_tokens": (
                    0 if generated is None else output_token_count(generated)
                ),
                "finish_reason": (
                    None if generated is None else output_finish_reason(generated)
                ),
                "stop_reason": stop_reason,
                "contains_think_close": "</think>" in generation,
                "generated_prediction": generated_prediction,
                "parse_error": (
                    bool(condition["generation"])
                    and generated_prediction is None
                ),
                "margin_prompt_sha256": sha256_text(margin_prompt),
                "score": score_from_output(margin, token_ids),
            }
            completed[row["id"]] = row
        atomic_write_jsonl(predictions_path, ordered_rows(records, completed))
        print(
            f"condition={condition_name} batch={batch_index} "
            f"complete={len(completed)}/{len(records)}",
            flush=True,
        )
    rows = ordered_rows(records, completed)
    if len(rows) != len(records):
        raise RuntimeError(f"condition {condition_name} is incomplete")
    summary = {
        "campaign_id": config["campaign_id"],
        "condition": condition,
        "config_sha256": config_sha256,
        "model": config["model"],
        "generation_settings": (
            config["generation"] if condition["generation"] else None
        ),
        "elapsed_seconds_this_invocation": time.time() - started,
        **summarize_predictions(rows),
    }
    atomic_write_json(output_dir / condition_name / "result.json", summary)
    macro = summary["metrics"]["macro"]["macro"]
    print(
        f"condition={condition_name} macro_auroc={macro['auroc']:.6f} "
        f"macro_brier={macro['brier']:.6f}",
        flush=True,
    )
    return rows


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    validate_config(config)
    records = load_jsonl(args.input)
    if len(records) != int(config["scope"]["rows"]):
        raise ValueError("input row count differs from the frozen config")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        records = records[: args.limit]
    if len({str(record["id"]) for record in records}) != len(records):
        raise ValueError("input contains duplicate IDs")
    config_sha256 = sha256_file(args.config)
    reasoning_instruction = REASONING_INSTRUCTION.read_text(encoding="utf-8")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model = config["model"]
    engine = config["engine"]
    tokenizer = AutoTokenizer.from_pretrained(
        model["id"],
        revision=model["revision"],
    )
    token_ids = binary_token_ids(tokenizer)
    generation = config["generation"]
    generation_sampling = SamplingParams(
        max_tokens=int(generation["max_tokens"]),
        min_tokens=int(generation["min_tokens"]),
        temperature=float(generation["temperature"]),
        top_p=float(generation["top_p"]),
        top_k=int(generation["top_k"]),
        min_p=float(generation["min_p"]),
        presence_penalty=float(generation["presence_penalty"]),
        repetition_penalty=float(generation["repetition_penalty"]),
        seed=int(generation["seed"]),
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
        tensor_parallel_size=1,
        gpu_memory_utilization=float(engine["gpu_memory_utilization"]),
        max_model_len=int(engine["max_model_len"]),
        max_num_seqs=int(engine["max_num_seqs"]),
        enable_prefix_caching=bool(engine["enable_prefix_caching"]),
        language_model_only=bool(engine["language_model_only"]),
        seed=int(generation["seed"]),
    )
    selected = set(args.condition or [])
    conditions = [
        condition
        for condition in config["conditions"]
        if not selected or condition["name"] in selected
    ]
    results: dict[str, list[dict[str, Any]]] = {}
    for condition in conditions:
        results[condition["name"]] = run_condition(
            llm,
            tokenizer,
            generation_sampling,
            margin_sampling,
            records,
            condition,
            reasoning_instruction,
            args.output,
            config,
            config_sha256,
            token_ids,
            force=args.force,
        )
    if len(results) == 3:
        comparisons = {
            "direct_to_visible_reasoning": paired_condition_comparison(
                results["direct"], results["visible_reasoning"]
            ),
            "direct_to_thinking_reasoning": paired_condition_comparison(
                results["direct"], results["thinking_reasoning"]
            ),
            "visible_to_thinking_reasoning": paired_condition_comparison(
                results["visible_reasoning"], results["thinking_reasoning"]
            ),
        }
        atomic_write_json(
            args.output / "comparison.json",
            {
                "campaign_id": config["campaign_id"],
                "config_sha256": config_sha256,
                "rows": len(records),
                "comparisons": comparisons,
            },
        )


if __name__ == "__main__":
    main()
