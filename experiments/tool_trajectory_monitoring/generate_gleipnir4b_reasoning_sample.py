#!/usr/bin/env python3
"""Generate paired visible reasoning from Gleipnir 4B and its base model."""

from __future__ import annotations

import argparse
import hashlib
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from experiments.tool_trajectory_monitoring import (
    benchmark_qwen_ordinal_reasoning_ood as ordinal_reasoning,
)
from experiments.tool_trajectory_monitoring.benchmark_gpt_oss_ood import (
    batches,
    ordered_predictions,
    validate_inputs,
)
from experiments.tool_trajectory_monitoring.benchmark_qwen_binary_reasoning_ood import (
    reasoned_prediction_row,
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
    "gleipnir4b_reasoning_sample.json"
)
DEFAULT_OUTPUT = Path(
    "results/tool_trajectory_monitoring/gleipnir4b_reasoning_sample"
)
CONDITIONS = ("base", "gleipnir")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stop-after-canary", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _stable_key(record_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}\0{record_id}".encode()).hexdigest()


def select_sample(
    records: list[dict[str, Any]], *, rows: int, seed: int
) -> list[dict[str, Any]]:
    """Select proportional source quotas by a label-blind stable hash ranking."""
    if rows < 1 or rows > len(records):
        raise ValueError("sample row count must lie within the input row count")
    identities = [str(record.get("id")) for record in records]
    if len(identities) != len(set(identities)):
        raise ValueError("cannot sample records with duplicate IDs")
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_source[str(record["metadata"]["source_dataset"])].append(record)
    if rows < len(by_source):
        raise ValueError("sample is too small to retain every OOD source")

    exact = {
        source: rows * len(source_records) / len(records)
        for source, source_records in by_source.items()
    }
    quotas = {source: max(1, math.floor(value)) for source, value in exact.items()}
    if sum(quotas.values()) > rows:
        raise ValueError("minimum source quotas exceed the sample size")
    remainder_order = sorted(
        by_source,
        key=lambda source: (-(exact[source] - math.floor(exact[source])), source),
    )
    for source in remainder_order[: rows - sum(quotas.values())]:
        quotas[source] += 1

    selected = []
    for source, source_records in by_source.items():
        ranked = sorted(
            source_records,
            key=lambda record: (
                _stable_key(str(record["id"]), seed),
                str(record["id"]),
            ),
        )
        selected.extend(ranked[: quotas[source]])
    return sorted(
        selected,
        key=lambda record: (
            _stable_key(str(record["id"]), seed),
            str(record["id"]),
        ),
    )


def selected_ids_sha256(records: list[dict[str, Any]]) -> str:
    payload = "\n".join(str(record["id"]) for record in records) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("status") != "frozen_generation":
        raise ValueError("reasoning sample config must be frozen before generation")
    model = config.get("model", {})
    if model.get("id") != "Qwen/Qwen3.5-4B":
        raise ValueError("reasoning comparison must use Qwen3.5-4B")
    prompt = config.get("prompt", {})
    if prompt.get("role") != "teacher" or prompt.get("enable_thinking") is not False:
        raise ValueError("reasoning comparison must use visible non-thinking review")
    if prompt.get("assistant_suffix") != QWEN_NON_THINKING_ASSISTANT_SUFFIX:
        raise ValueError("Qwen non-thinking assistant boundary drifted")
    instruction_path = Path(str(prompt.get("instruction_file", "")))
    if not instruction_path.is_file():
        raise ValueError("visible-reasoning instruction is missing")
    if sha256_file(instruction_path) != prompt.get("instruction_sha256"):
        raise ValueError("visible-reasoning instruction checksum drifted")
    generation = config.get("generation", {})
    if generation.get("stop") != "\nPrediction:":
        raise ValueError("visible reasoning must terminate at Prediction:")
    if generation.get("include_stop_str_in_output") is not True:
        raise ValueError("terminal Prediction: boundary must be retained")
    if float(generation.get("temperature", -1)) != 0.0:
        raise ValueError("paired reasoning must use greedy decoding")
    if config.get("scoring", {}).get("tokens") != ["0", "1"]:
        raise ValueError("terminal tokens must remain literal 0 and 1")

    scope = config.get("scope", {})
    sample_rows = int(scope.get("sample_rows", 0))
    total_rows = int(scope.get("rows", 0))
    if sample_rows < 1 or sample_rows > total_rows:
        raise ValueError("invalid frozen sample size")
    selection = config.get("selection", {})
    if selection.get("method") != "proportional_source_quota_then_sha256_rank":
        raise ValueError("sample selection method drifted")
    if not selection.get("selected_ids_sha256"):
        raise ValueError("frozen sample identity checksum is missing")

    gleipnir = config.get("gleipnir", {})
    if (
        gleipnir.get("job_name")
        != "soft-n21837-mixed-qwen35-4b-seed0"
        or gleipnir.get("target") != "kimi_soft"
        or int(gleipnir.get("train_rows", 0)) != 21_837
        or float(gleipnir.get("soft_loss_weight", -1)) != 1.0
        or float(gleipnir.get("direct_loss_weight", -1)) != 0.0
        or int(gleipnir.get("rank", 0)) != 128
    ):
        raise ValueError("Gleipnir must be the frozen mixed-data soft-only 4B adapter")
    engine = config.get("engine", {})
    if int(engine.get("max_lora_rank", 0)) != 128:
        raise ValueError("vLLM LoRA rank differs from the frozen adapter")
    audited_max = int(engine.get("audited_max_prompt_tokens", 0))
    sample_max = int(engine.get("audited_sample_max_prompt_tokens", 0))
    sample_total = int(
        engine.get("audited_sample_prompt_tokens_per_condition", 0)
    )
    if sample_max < 1 or sample_max > audited_max or sample_total < sample_max:
        raise ValueError("frozen sampled prompt-token audit is invalid")
    max_tokens = int(generation.get("max_tokens", 0))
    if audited_max + max_tokens + 1 > int(engine.get("max_model_len", 0)):
        raise ValueError(
            "engine context does not cover prompt, rationale, and decision"
        )
    if int(engine.get("canary_rows", 0)) < 1:
        raise ValueError("technical canary must contain at least one sampled row")


def validate_artifacts(config: dict[str, Any]) -> None:
    gleipnir = config["gleipnir"]
    adapter_path = Path(gleipnir["serving_adapter"])
    adapter_weights = adapter_path / "adapter_model.safetensors"
    if sha256_file(adapter_weights) != gleipnir["serving_adapter_sha256"]:
        raise ValueError("Gleipnir serving adapter checksum drifted")
    metadata_path = Path(gleipnir["training_metadata"])
    if sha256_file(metadata_path) != gleipnir["training_metadata_sha256"]:
        raise ValueError("Gleipnir training metadata checksum drifted")
    metadata = load_json(metadata_path)
    if (
        metadata.get("model") != config["model"]["id"]
        or metadata.get("model_revision") != config["model"]["revision"]
        or metadata.get("finetuning_mode") != "lora"
        or metadata.get("decision_head_mode") != "token_logits"
    ):
        raise ValueError("Gleipnir training metadata identity drifted")
    parity_path = Path(gleipnir["serving_parity"])
    if sha256_file(parity_path) != gleipnir["serving_parity_sha256"]:
        raise ValueError("Gleipnir serving parity checksum drifted")
    parity = load_json(parity_path)
    if (
        parity.get("passed") is not True
        or parity.get("model_size") != "4b"
        or parity.get("job_name") != gleipnir["job_name"]
    ):
        raise ValueError("Gleipnir serving parity gate did not pass")


def condition_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty condition")
    return {
        "rows": len(rows),
        "decisions": dict(Counter(str(row["prediction"]) for row in rows)),
        "prompt_tokens": {
            "total": sum(int(row["prompt_tokens"]) for row in rows),
            "min": min(int(row["prompt_tokens"]) for row in rows),
            "max": max(int(row["prompt_tokens"]) for row in rows),
        },
        "rationale_tokens": {
            "total": sum(int(row["rationale_tokens"]) for row in rows),
            "min": min(int(row["rationale_tokens"]) for row in rows),
            "max": max(int(row["rationale_tokens"]) for row in rows),
        },
        "boundary_recoveries": dict(
            Counter(str(row["boundary_recovery"]) for row in rows)
        ),
        "unique_rationale_hashes": len(
            {str(row["rationale_sha256"]) for row in rows}
        ),
    }


def paired_summary(
    base_rows: list[dict[str, Any]], gleipnir_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    base = {str(row["id"]): row for row in base_rows}
    gleipnir = {str(row["id"]): row for row in gleipnir_rows}
    if base.keys() != gleipnir.keys() or len(base) != len(base_rows):
        raise ValueError("paired condition row identities differ")
    exact = sum(
        str(base[row_id]["rationale"]) == str(gleipnir[row_id]["rationale"])
        for row_id in base
    )
    decisions = sum(
        int(base[row_id]["prediction"]) == int(gleipnir[row_id]["prediction"])
        for row_id in base
    )
    return {
        "rows": len(base),
        "exact_rationale_matches": exact,
        "changed_rationales": len(base) - exact,
        "decision_agreements": decisions,
        "decision_disagreements": len(base) - decisions,
    }


def blinded_pair_rows(
    records: list[dict[str, Any]],
    base_rows: list[dict[str, Any]],
    gleipnir_rows: list[dict[str, Any]],
    *,
    blind_seed: int,
    binary_cache_prefix: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base = {str(row["id"]): row for row in base_rows}
    gleipnir = {str(row["id"]): row for row in gleipnir_rows}
    judge_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for record in records:
        record_id = str(record["id"])
        if record_id not in base or record_id not in gleipnir:
            raise ValueError(f"missing paired response for id={record_id!r}")
        swap = int(_stable_key(record_id, blind_seed)[:2], 16) % 2 == 1
        condition_a, condition_b = (
            ("gleipnir", "base") if swap else ("base", "gleipnir")
        )
        by_condition = {"base": base[record_id], "gleipnir": gleipnir[record_id]}
        candidates = {}
        for slot, condition in (("a", condition_a), ("b", condition_b)):
            row = by_condition[condition]
            candidates[f"candidate_{slot}"] = (
                f"{row['rationale']}\nPrediction:{row['prediction']}"
            )
        judge_rows.append(
            {
                "pair_id": record_id,
                "trajectory": extract_trajectory(
                    str(record["prompt"]), binary_cache_prefix
                ),
                **candidates,
            }
        )
        metadata = record["metadata"]
        key_rows.append(
            {
                "pair_id": record_id,
                "candidate_a_condition": condition_a,
                "candidate_b_condition": condition_b,
                "source": str(metadata["source_dataset"]),
                "ground_truth": int(metadata["ground_truth"]),
            }
        )
    return judge_rows, key_rows


def selection_manifest(
    records: list[dict[str, Any]], config: dict[str, Any], config_sha256: str
) -> dict[str, Any]:
    source_counts = Counter(
        str(record["metadata"]["source_dataset"]) for record in records
    )
    label_counts = Counter(
        str(int(record["metadata"]["ground_truth"])) for record in records
    )
    return {
        "campaign_id": config["campaign_id"],
        "config_sha256": config_sha256,
        "input": config["scope"]["input"],
        "input_sha256": config["scope"]["input_sha256"],
        "selection": config["selection"],
        "rows": len(records),
        "fraction_of_ood": len(records) / int(config["scope"]["rows"]),
        "selected_ids": [str(record["id"]) for record in records],
        "selected_ids_sha256": selected_ids_sha256(records),
        "source_counts": dict(source_counts),
        "label_counts_audit_only": dict(label_counts),
    }


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    validate_config(config)
    validate_artifacts(config)
    all_records = validate_inputs(
        {
            **config,
            "prompt": {
                "prompt_set_id": config["prompt"]["binary_prompt_set_id"],
                "template_sha256": config["prompt"]["binary_template_sha256"],
            },
        }
    )
    records = select_sample(
        all_records,
        rows=int(config["scope"]["sample_rows"]),
        seed=int(config["selection"]["seed"]),
    )
    if selected_ids_sha256(records) != config["selection"]["selected_ids_sha256"]:
        raise ValueError("deterministic sample identity checksum drifted")
    config_sha256 = sha256_file(args.config)

    prompt_set = load_prompt_set()
    binary_template = prompt_set.teacher
    if binary_template.template_sha256 != config["prompt"]["binary_template_sha256"]:
        raise ValueError("working-tree binary teacher prompt differs")
    instruction = Path(config["prompt"]["instruction_file"]).read_text(
        encoding="utf-8"
    )

    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    model = config["model"]
    engine = config["engine"]
    generation = config["generation"]
    tokenizer = AutoTokenizer.from_pretrained(
        model["id"], revision=model["revision"]
    )
    token_ids = binary_token_ids(tokenizer)
    generation_prompts = []
    for record in records:
        trajectory = extract_trajectory(
            str(record["prompt"]), binary_template.cache_prefix
        )
        user_prompt = render_ordinal_user_prompt(instruction, trajectory)
        generation_prompts.append(
            render_generation_prompt(
                tokenizer,
                user_prompt,
                enable_thinking=bool(config["prompt"]["enable_thinking"]),
                assistant_suffix=str(config["prompt"]["assistant_suffix"]),
            )
        )
    prompt_tokens = [
        len(tokenizer.encode(prompt, add_special_tokens=False))
        for prompt in generation_prompts
    ]
    if sum(prompt_tokens) != int(
        engine["audited_sample_prompt_tokens_per_condition"]
    ) or max(prompt_tokens) != int(engine["audited_sample_max_prompt_tokens"]):
        raise ValueError("sampled prompt-token audit drifted")
    if max(prompt_tokens) + int(generation["max_tokens"]) + 1 > int(
        engine["max_model_len"]
    ):
        raise ValueError("selected prompt can reach the engine context limit")

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
        enable_lora=True,
        max_lora_rank=int(engine["max_lora_rank"]),
        max_loras=int(engine["max_loras"]),
        seed=int(engine["seed"]),
    )
    request = LoRARequest(
        "gleipnir4b_mixed_soft",
        1,
        str(config["gleipnir"]["serving_adapter"]),
    )
    requests = {"base": None, "gleipnir": request}
    prompt_by_id = dict(
        zip((str(record["id"]) for record in records), generation_prompts, strict=True)
    )
    completed_by_condition: dict[str, dict[str, dict[str, Any]]] = {}
    for condition in CONDITIONS:
        path = args.output / condition / "predictions.jsonl"
        existing = [] if args.force or not path.is_file() else load_jsonl(path)
        completed = ordinal_reasoning.validate_existing(
            existing, records, config_sha256
        )
        if any(row.get("condition") != condition for row in completed.values()):
            raise ValueError(f"resumable {condition} condition identity drifted")
        completed_by_condition[condition] = completed

    def score_rows(condition: str, requested_records: list[dict[str, Any]]) -> None:
        completed = completed_by_condition[condition]
        pending = [
            record
            for record in requested_records
            if str(record["id"]) not in completed
        ]
        for batch_index, batch in enumerate(
            batches(pending, int(engine["batch_rows"])), start=1
        ):
            prompts = [prompt_by_id[str(record["id"])] for record in batch]
            rationale_outputs = llm.generate(
                prompts,
                reasoning_sampling,
                lora_request=requests[condition],
            )
            continuations = [
                ordinal_reasoning.reasoning_continuation(
                    output, str(generation["stop"])
                )[0]
                for output in rationale_outputs
            ]
            margin_prompts = [
                prompt + continuation
                for prompt, continuation in zip(prompts, continuations, strict=True)
            ]
            margin_outputs = llm.generate(
                margin_prompts,
                margin_sampling,
                lora_request=requests[condition],
            )
            if len(rationale_outputs) != len(batch) or len(margin_outputs) != len(
                batch
            ):
                raise RuntimeError(
                    "vLLM output count differs from paired request count"
                )
            for record, prompt, rationale, margin_prompt, margin_output in zip(
                batch,
                prompts,
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
                packed["condition"] = condition
                completed[packed["id"]] = packed
            atomic_write_jsonl(
                args.output / condition / "predictions.jsonl",
                ordered_predictions(records, completed),
            )
            print(
                f"{condition}: batch={batch_index} "
                f"complete={len(completed)}/{len(records)}",
                flush=True,
            )

    started = time.time()
    canary_records = records[: int(engine["canary_rows"])]
    for condition in CONDITIONS:
        score_rows(condition, canary_records)
    canary_rows = {
        condition: ordered_predictions(
            canary_records, completed_by_condition[condition]
        )
        for condition in CONDITIONS
    }
    if any(
        len(rows) != len(canary_records)
        or not all(math.isfinite(float(row["score"])) for row in rows)
        for rows in canary_rows.values()
    ):
        raise RuntimeError("paired visible-reasoning canary failed")
    atomic_write_json(
        args.output / "canary_result.json",
        {
            "campaign_id": config["campaign_id"],
            "config_sha256": config_sha256,
            "base": condition_summary(canary_rows["base"]),
            "gleipnir": condition_summary(canary_rows["gleipnir"]),
            "paired": paired_summary(
                canary_rows["base"], canary_rows["gleipnir"]
            ),
        },
    )
    print(f"paired canary complete rows={len(canary_records)}", flush=True)
    if args.stop_after_canary:
        return

    for condition in CONDITIONS:
        score_rows(condition, records)
    predictions = {
        condition: ordered_predictions(records, completed_by_condition[condition])
        for condition in CONDITIONS
    }
    if any(
        len(rows) != len(records)
        or not all(math.isfinite(float(row["score"])) for row in rows)
        for rows in predictions.values()
    ):
        raise RuntimeError("paired visible-reasoning sample is incomplete")

    judge_rows, key_rows = blinded_pair_rows(
        records,
        predictions["base"],
        predictions["gleipnir"],
        blind_seed=int(config["blinding"]["seed"]),
        binary_cache_prefix=binary_template.cache_prefix,
    )
    atomic_write_jsonl(args.output / "judge_inputs.jsonl", judge_rows)
    atomic_write_jsonl(args.output / "judge_key.jsonl", key_rows)
    manifest = selection_manifest(records, config, config_sha256)
    manifest["prompt_tokens"] = {
        "total_per_condition": sum(prompt_tokens),
        "total_both_conditions": 2 * sum(prompt_tokens),
        "min": min(prompt_tokens),
        "max": max(prompt_tokens),
        "mean": sum(prompt_tokens) / len(prompt_tokens),
    }
    atomic_write_json(args.output / "selection_manifest.json", manifest)
    result = {
        "campaign_id": config["campaign_id"],
        "config_sha256": config_sha256,
        "model": model,
        "gleipnir": config["gleipnir"],
        "prompt": config["prompt"],
        "generation": generation,
        "selection": manifest,
        "base": condition_summary(predictions["base"]),
        "gleipnir_condition": condition_summary(predictions["gleipnir"]),
        "paired": paired_summary(
            predictions["base"], predictions["gleipnir"]
        ),
        "runtime": {
            "vllm_version": vllm.__version__,
            "torch_version": torch.__version__,
            "gpu_name": torch.cuda.get_device_name(0),
            "elapsed_seconds_this_invocation": time.time() - started,
        },
        "artifacts": {
            "base_predictions": str(args.output / "base/predictions.jsonl"),
            "gleipnir_predictions": str(
                args.output / "gleipnir/predictions.jsonl"
            ),
            "blinded_judge_inputs": str(args.output / "judge_inputs.jsonl"),
            "blinding_key": str(args.output / "judge_key.jsonl"),
        },
    }
    atomic_write_json(args.output / "result.json", result)
    print(
        f"complete rows={len(records)} "
        f"changed_rationales={result['paired']['changed_rationales']} "
        f"decision_disagreements={result['paired']['decision_disagreements']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
