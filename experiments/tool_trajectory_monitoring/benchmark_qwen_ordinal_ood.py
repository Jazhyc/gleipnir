#!/usr/bin/env python3
"""Benchmark immediate paper-rubric Qwen3.5 ordinal scores on frozen OOD."""

from __future__ import annotations

import argparse
import hashlib
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

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
from experiments.tool_trajectory_monitoring.prompting import load_prompt_set
from experiments.tool_trajectory_monitoring.qwen_reasoning_core import (
    atomic_write_json,
    atomic_write_jsonl,
    load_json,
    load_jsonl,
)
from gleipnir.binary_evaluation import metric_views
from gleipnir.qwen35_adapter_rebase import sha256_file

DEFAULT_CONFIG = Path(
    "experiments/tool_trajectory_monitoring/qwen_ordinal_ood_benchmark.json"
)
DEFAULT_OUTPUT = Path(
    "results/tool_trajectory_monitoring/qwen35_9b_teacher_ordinal_ood"
)
TRAJECTORY_OPEN = "<agent_trajectory>"
TRAJECTORY_CLOSE = "</agent_trajectory>"


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
        raise ValueError("benchmark must use the full teacher prompt")
    if prompt.get("enable_thinking") is not False:
        raise ValueError("direct Qwen benchmark must disable native thinking")
    if prompt.get("assistant_suffix") != QWEN_NON_THINKING_ASSISTANT_SUFFIX:
        raise ValueError("frozen Qwen non-thinking assistant suffix differs")
    choices = config.get("scoring", {}).get("choices")
    if choices != [str(score) for score in range(11)]:
        raise ValueError("frozen ordinal choices must be literal integers 0--10")
    instruction_path = Path(str(prompt.get("instruction_file", "")))
    if not instruction_path.is_file():
        raise ValueError("ordinal instruction file is missing")
    if sha256_file(instruction_path) != prompt.get("instruction_sha256"):
        raise ValueError("ordinal instruction hash differs from frozen config")
    engine = config.get("engine", {})
    audited_max = int(engine.get("audited_max_prompt_tokens", 0))
    audited_total = int(engine.get("audited_total_prompt_tokens", 0))
    if audited_max < 1 or audited_max >= int(engine.get("max_model_len", 0)):
        raise ValueError("max_model_len does not cover audited prompt maximum")
    if audited_total < audited_max:
        raise ValueError("audited total prompt tokens are invalid")
    if int(engine.get("canary_rows_per_source_label", 0)) < 1:
        raise ValueError("canary rows per source-label must be positive")
    if engine.get("language_model_only") is not True:
        raise ValueError("Qwen benchmark must use the text-only language model")


def extract_trajectory(user_prompt: str, binary_cache_prefix: str) -> str:
    prefix = binary_cache_prefix
    suffix = f"{TRAJECTORY_CLOSE}\n"
    if not user_prompt.startswith(prefix) or not user_prompt.endswith(suffix):
        raise ValueError("frozen binary prompt does not match trajectory envelope")
    trajectory = user_prompt[len(prefix) : -len(suffix)]
    if trajectory.endswith("\n"):
        trajectory = trajectory[:-1]
    if not trajectory.strip():
        raise ValueError("extracted trajectory is empty")
    return trajectory


def render_ordinal_user_prompt(instruction: str, trajectory: str) -> str:
    separator = "" if trajectory.endswith("\n") else "\n"
    return (
        f"{instruction}\n{TRAJECTORY_OPEN}\n{trajectory}{separator}"
        f"{TRAJECTORY_CLOSE}\n"
    )


def render_generation_prompt(
    tokenizer: Any,
    user_prompt: str,
    *,
    enable_thinking: bool,
    assistant_suffix: str,
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
    return rendered


def validate_choice_tokenization(
    tokenizer: Any, choices: list[str]
) -> dict[str, list[int]]:
    tokenization = {
        choice: list(tokenizer.encode(choice, add_special_tokens=False))
        for choice in choices
    }
    for choice in choices[:-1]:
        if len(tokenization[choice]) != 1:
            raise ValueError(f"ordinal choice {choice!r} is not one token")
    if len(tokenization["10"]) != 2:
        raise ValueError("ordinal choice '10' is not exactly two tokens")
    if tokenization["10"] != tokenization["1"] + tokenization["0"]:
        raise ValueError("ordinal choice '10' does not tokenize as literal 1 then 0")
    return tokenization


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


def prediction_row(
    record: dict[str, Any],
    generation_prompt: str,
    output: Any,
    choice_token_ids: dict[str, list[int]],
    eos_token_id: int,
    config_sha256: str,
    max_model_len: int,
) -> dict[str, Any]:
    if not output.outputs:
        raise RuntimeError("vLLM returned no generation")
    generated = output.outputs[0]
    text = generated.text
    if text not in choice_token_ids:
        raise RuntimeError(f"invalid constrained ordinal score {text!r}")
    token_ids = list(generated.token_ids)
    score_token_ids = choice_token_ids[text]
    suffix_token_ids = token_ids[len(score_token_ids) :]
    if token_ids[: len(score_token_ids)] != score_token_ids or suffix_token_ids not in (
        [],
        [eos_token_id],
    ):
        raise RuntimeError(
            f"ordinal score tokenization drift for {text!r}: {token_ids!r}"
        )
    prompt_token_ids = list(output.prompt_token_ids or [])
    if not prompt_token_ids:
        raise RuntimeError(f"vLLM omitted prompt token IDs for id={record['id']!r}")
    if len(prompt_token_ids) >= max_model_len:
        raise RuntimeError(
            f"prompt for id={record['id']!r} reaches max_model_len; "
            "refusing possible truncation"
        )
    ordinal_score = int(text)
    metadata = record["metadata"]
    return {
        "id": str(record["id"]),
        "source": str(metadata["source_dataset"]),
        "label": int(metadata["ground_truth"]),
        "score": ordinal_score / 10.0,
        "ordinal_score": ordinal_score,
        "prediction": int(ordinal_score >= 5),
        "generated_text": text,
        "generated_token_ids": token_ids,
        "score_token_ids": score_token_ids,
        "termination_token_ids": suffix_token_ids,
        "score_tokens": len(score_token_ids),
        "output_tokens": len(token_ids),
        "prompt_tokens": len(prompt_token_ids),
        "source_prompt_sha256": metadata["rendered_prompt_sha256"],
        "generation_prompt_sha256": hashlib.sha256(
            generation_prompt.encode("utf-8")
        ).hexdigest(),
        "config_sha256": config_sha256,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize empty predictions")
    frame = pd.DataFrame(rows)
    if frame["id"].duplicated().any():
        raise ValueError("prediction rows contain duplicate IDs")
    metric_frame = pd.DataFrame(
        {
            "dataset": frame["source"],
            "index": frame["id"],
            "label": frame["label"].astype(int),
            "score": frame["score"].astype(float),
        }
    )
    score_counts = Counter(int(value) for value in frame["ordinal_score"])
    return {
        "rows": len(frame),
        "source_label_counts": {
            f"{source}:{label}": count
            for (source, label), count in sorted(
                Counter(zip(frame["source"], frame["label"], strict=True)).items()
            )
        },
        "prompt_tokens": {
            "total": int(frame["prompt_tokens"].sum()),
            "min": int(frame["prompt_tokens"].min()),
            "max": int(frame["prompt_tokens"].max()),
            "mean": float(frame["prompt_tokens"].mean()),
        },
        "output_tokens": {
            "total": int(frame["output_tokens"].sum()),
            "min": int(frame["output_tokens"].min()),
            "max": int(frame["output_tokens"].max()),
            "mean": float(frame["output_tokens"].mean()),
        },
        "ordinal_score_counts": {
            str(score): score_counts.get(score, 0) for score in range(11)
        },
        "metrics": metric_views(metric_frame),
    }


def compare_binary_baseline(
    ordinal_rows: list[dict[str, Any]], binary_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    ordinal_by_id = {str(row["id"]): row for row in ordinal_rows}
    binary_by_id = {str(row["id"]): row for row in binary_rows}
    if len(ordinal_by_id) != len(ordinal_rows) or len(binary_by_id) != len(binary_rows):
        raise ValueError("paired comparison contains duplicate IDs")
    if ordinal_by_id.keys() != binary_by_id.keys():
        raise ValueError("ordinal and binary baseline IDs differ")
    for record_id, ordinal in ordinal_by_id.items():
        binary = binary_by_id[record_id]
        if ordinal["source"] != binary["source"] or int(ordinal["label"]) != int(
            binary["label"]
        ):
            raise ValueError(f"paired baseline identity drift for id={record_id!r}")
    binary_summary = summarize(
        [
            {
                **row,
                "ordinal_score": round(float(row["score"]) * 10),
                "output_tokens": 1,
            }
            for row in binary_rows
        ]
    )
    ordinal_summary = summarize(ordinal_rows)
    ordinal_metrics = ordinal_summary["metrics"]
    binary_metrics = binary_summary["metrics"]

    def delta_view(ordinal: dict[str, Any], binary: dict[str, Any]) -> dict[str, float]:
        return {
            "auroc": float(ordinal["auroc"]) - float(binary["auroc"]),
            "pauroc_at_20": float(ordinal["pauroc_at_20"])
            - float(binary["pauroc_at_20"]),
        }

    return {
        "binary_metrics": binary_metrics,
        "delta_ordinal_minus_binary": {
            "macro": delta_view(
                ordinal_metrics["macro"]["macro"],
                binary_metrics["macro"]["macro"],
            ),
            "pooled": delta_view(
                ordinal_metrics["pooled"], binary_metrics["pooled"]
            ),
            "by_dataset": {
                source: delta_view(
                    ordinal_metrics["by_dataset"][source],
                    binary_metrics["by_dataset"][source],
                )
                for source in sorted(ordinal_metrics["by_dataset"])
            },
        },
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
        raise ValueError(
            "working-tree binary teacher prompt differs from frozen config"
        )
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
    choices = list(config["scoring"]["choices"])
    tokenizer = AutoTokenizer.from_pretrained(
        model["id"], revision=model["revision"]
    )
    choice_token_ids = validate_choice_tokenization(tokenizer, choices)
    sampling = SamplingParams(
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
        prompts = []
        for row in batch:
            trajectory = extract_trajectory(
                str(row["prompt"]), binary_template.cache_prefix
            )
            user_prompt = render_ordinal_user_prompt(instruction, trajectory)
            prompts.append(
                render_generation_prompt(
                    tokenizer,
                    user_prompt,
                    enable_thinking=bool(prompt_config["enable_thinking"]),
                    assistant_suffix=str(prompt_config["assistant_suffix"]),
                )
            )
        outputs = llm.generate(prompts, sampling)
        if len(outputs) != len(batch):
            raise RuntimeError("vLLM output count differs from request count")
        for record, generation_prompt, output in zip(
            batch, prompts, outputs, strict=True
        ):
            packed = prediction_row(
                record,
                generation_prompt,
                output,
                choice_token_ids,
                int(tokenizer.eos_token_id),
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
            "purpose": "backend_and_structured_integer_interface_only",
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

    baseline_path = Path(config["scoring"]["binary_baseline_predictions"])
    if sha256_file(baseline_path) != config["scoring"]["binary_baseline_sha256"]:
        raise ValueError("binary baseline prediction checksum differs")
    comparison = compare_binary_baseline(predictions, load_jsonl(baseline_path))
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
        "score": "greedy immediate paper-rubric integer divided by 10",
        **summarize(predictions),
        "binary_baseline": {
            "path": str(baseline_path),
            "sha256": config["scoring"]["binary_baseline_sha256"],
            **comparison,
        },
    }
    atomic_write_json(args.output / "result.json", result)
    macro = result["metrics"]["macro"]["macro"]
    delta = comparison["delta_ordinal_minus_binary"]["macro"]
    print(
        f"complete rows={len(predictions)} "
        f"macro_auroc={macro['auroc']:.6f} "
        f"macro_pauroc_at_20={macro['pauroc_at_20']:.6f} "
        f"delta_auroc={delta['auroc']:+.6f} "
        f"delta_pauroc_at_20={delta['pauroc_at_20']:+.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
