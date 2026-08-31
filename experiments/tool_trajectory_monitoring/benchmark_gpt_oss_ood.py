#!/usr/bin/env python3
"""Benchmark GPT-OSS-120B direct binary logits on the frozen OOD suite."""

from __future__ import annotations

import argparse
import hashlib
import math
import time
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.tool_trajectory_monitoring.prompting import load_prompt_set
from experiments.tool_trajectory_monitoring.qwen_reasoning_core import (
    atomic_write_json,
    atomic_write_jsonl,
    load_json,
    load_jsonl,
)
from gleipnir.binary_evaluation import binary_token_ids, logprob_value, metric_views
from gleipnir.qwen35_adapter_rebase import sha256_file

DEFAULT_CONFIG = Path(
    "experiments/tool_trajectory_monitoring/gpt_oss_ood_benchmark.json"
)
DEFAULT_OUTPUT = Path("results/tool_trajectory_monitoring/gpt_oss_120b_ood")
HARMONY_ASSISTANT_START = "<|start|>assistant"


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
    if config.get("model", {}).get("id") != "openai/gpt-oss-120b":
        raise ValueError("frozen model must be openai/gpt-oss-120b")
    prompt = config.get("prompt", {})
    if prompt.get("role") != "teacher":
        raise ValueError("benchmark must use the full teacher prompt")
    if prompt.get("assistant_boundary") != "<|channel|>final<|message|>":
        raise ValueError("benchmark must score at the Harmony final channel")
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


def validate_inputs(config: dict[str, Any]) -> list[dict[str, Any]]:
    scope = config["scope"]
    input_path = Path(scope["input"])
    manifest_path = Path(scope["manifest"])
    if sha256_file(input_path) != scope["input_sha256"]:
        raise ValueError("OOD input checksum differs from the frozen config")
    manifest = load_json(manifest_path)
    if manifest.get("output", {}).get("sha256") != scope["input_sha256"]:
        raise ValueError("OOD manifest and benchmark config disagree on input hash")
    if int(manifest.get("output", {}).get("rows", -1)) != int(scope["rows"]):
        raise ValueError("OOD manifest and benchmark config disagree on row count")
    prompt = config["prompt"]
    manifest_prompt = manifest.get("prompt", {})
    if manifest_prompt.get("prompt_set_id") != prompt["prompt_set_id"]:
        raise ValueError("OOD prompt-set ID differs from the frozen config")
    if manifest_prompt.get("template_sha256") != prompt["template_sha256"]:
        raise ValueError("OOD teacher prompt hash differs from the frozen config")

    rows = load_jsonl(input_path)
    if len(rows) != int(scope["rows"]):
        raise ValueError("OOD input row count differs from the frozen config")
    ids = [str(row.get("id")) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("OOD input contains duplicate IDs")
    expected_sources = set(scope["sources"])
    actual_sources = {
        str(row.get("metadata", {}).get("source_dataset")) for row in rows
    }
    if actual_sources != expected_sources:
        raise ValueError("OOD source membership differs from the frozen config")
    for row in rows:
        metadata = row.get("metadata", {})
        if metadata.get("prompt_template_sha256") != prompt["template_sha256"]:
            raise ValueError(f"prompt hash drift for id={row.get('id')!r}")
        if int(metadata.get("ground_truth", -1)) not in {0, 1}:
            raise ValueError(f"non-binary label for id={row.get('id')!r}")
        value = row.get("prompt")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"empty prompt for id={row.get('id')!r}")
    return rows


def render_margin_prompt(
    tokenizer: Any,
    user_prompt: str,
    *,
    reasoning_effort: str,
    assistant_boundary: str,
    decision_prefix: str,
) -> str:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort=reasoning_effort,
    )
    if not rendered.endswith(HARMONY_ASSISTANT_START):
        raise ValueError(
            "GPT-OSS chat template no longer ends at the expected assistant start"
        )
    return f"{rendered}{assistant_boundary}{decision_prefix}"


def balanced_canary_rows(
    rows: list[dict[str, Any]],
    sources: list[str],
    *,
    rows_per_source_label: int,
) -> list[dict[str, Any]]:
    selected = []
    for source in sources:
        for label in (0, 1):
            eligible = [
                row
                for row in rows
                if row["metadata"]["source_dataset"] == source
                and int(row["metadata"]["ground_truth"]) == label
            ]
            if len(eligible) < rows_per_source_label:
                raise ValueError(
                    f"source={source!r} label={label} has too few canary rows"
                )
            selected.extend(eligible[:rows_per_source_label])
    return selected


def binary_logprobs(output: Any, token_ids: list[int]) -> tuple[float, float]:
    if not output.outputs or not output.outputs[0].logprobs:
        raise RuntimeError("vLLM returned no first-token logprobs")
    values = output.outputs[0].logprobs[0] or {}
    expanded = {int(key): logprob_value(value) for key, value in values.items()}
    missing = [token_id for token_id in token_ids if token_id not in expanded]
    if missing:
        raise RuntimeError(f"vLLM omitted requested token logprobs: {missing}")
    return expanded[token_ids[0]], expanded[token_ids[1]]


def normalized_score(logprob_0: float, logprob_1: float) -> float:
    difference = max(-80.0, min(80.0, logprob_1 - logprob_0))
    return 1.0 / (1.0 + math.exp(-difference))


def validate_existing(
    existing: list[dict[str, Any]],
    records: list[dict[str, Any]],
    config_sha256: str,
) -> dict[str, dict[str, Any]]:
    allowed = {str(row["id"]): row for row in records}
    completed = {}
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


def ordered_predictions(
    records: list[dict[str, Any]], completed: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    return [completed[str(row["id"])] for row in records if str(row["id"]) in completed]


def batches(rows: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    if size < 1:
        raise ValueError("batch size must be positive")
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def prediction_row(
    record: dict[str, Any],
    margin_prompt: str,
    output: Any,
    token_ids: list[int],
    config_sha256: str,
    max_model_len: int,
) -> dict[str, Any]:
    logprob_0, logprob_1 = binary_logprobs(output, token_ids)
    score = normalized_score(logprob_0, logprob_1)
    prompt_token_ids = list(output.prompt_token_ids or [])
    if not prompt_token_ids:
        raise RuntimeError(f"vLLM omitted prompt token IDs for id={record['id']!r}")
    if len(prompt_token_ids) >= max_model_len:
        raise RuntimeError(
            f"prompt for id={record['id']!r} reaches max_model_len; "
            "refusing possible truncation"
        )
    generated = list(output.outputs[0].token_ids)
    if len(generated) != 1 or generated[0] not in token_ids:
        raise RuntimeError(
            f"invalid constrained decision token for id={record['id']!r}"
        )
    metadata = record["metadata"]
    return {
        "id": str(record["id"]),
        "source": str(metadata["source_dataset"]),
        "label": int(metadata["ground_truth"]),
        "score": score,
        "prediction": int(generated[0] == token_ids[1]),
        "logprob_0": logprob_0,
        "logprob_1": logprob_1,
        "logit_margin_1_minus_0": logprob_1 - logprob_0,
        "token_id_0": token_ids[0],
        "token_id_1": token_ids[1],
        "prompt_tokens": len(prompt_token_ids),
        "source_prompt_sha256": metadata["rendered_prompt_sha256"],
        "margin_prompt_sha256": hashlib.sha256(
            margin_prompt.encode("utf-8")
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
        "metrics": metric_views(metric_frame),
    }


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
                reasoning_effort=prompt_config["reasoning_effort_header"],
                assistant_boundary=prompt_config["assistant_boundary"],
                decision_prefix=prompt_config["decision_prefix"],
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
        "score": "normalized direct final-channel probability for literal 1 versus 0",
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
