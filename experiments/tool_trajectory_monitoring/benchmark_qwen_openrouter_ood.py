#!/usr/bin/env python3
"""Benchmark exact one-token Qwen logits through OpenRouter raw completions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import requests
from dotenv import load_dotenv

from experiments.tool_trajectory_monitoring.benchmark_distilled_ood import (
    validate_inputs,
)
from experiments.tool_trajectory_monitoring.benchmark_gpt_oss_ood import (
    balanced_canary_rows,
    batches,
    normalized_score,
    ordered_predictions,
    summarize,
)
from experiments.tool_trajectory_monitoring.benchmark_qwen_ood import (
    QWEN_NON_THINKING_ASSISTANT_SUFFIX,
    render_margin_prompt,
)
from experiments.tool_trajectory_monitoring.prompting import load_prompt_set
from experiments.tool_trajectory_monitoring.qwen_reasoning_core import (
    atomic_write_json,
    atomic_write_jsonl,
    load_json,
    load_jsonl,
)
from gleipnir.openrouter import RETRYABLE_STATUS_CODES, retry_delay_seconds
from gleipnir.qwen35_adapter_rebase import sha256_file

DEFAULT_CONFIG = Path(
    "experiments/tool_trajectory_monitoring/"
    "qwen27b_compact_openrouter_ood_benchmark.json"
)
DEFAULT_OUTPUT = Path(
    "results/tool_trajectory_monitoring/qwen35_27b_compact_openrouter_ood"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stop-after-canary", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--concurrency",
        type=int,
        help="Operational worker override; does not change request semantics.",
    )
    parser.add_argument(
        "--request-start-interval-seconds",
        type=float,
        default=0.0,
        help="Minimum spacing between request starts across all workers.",
    )
    parser.add_argument(
        "--max-new-rows",
        type=int,
        help="Bound a resumable provider-health probe to this many new rows.",
    )
    parser.add_argument(
        "--request-batch-rows",
        type=int,
        help="Operational checkpoint-size override.",
    )
    parser.add_argument(
        "--provider-cooldown-seconds",
        type=float,
        default=0.0,
        help="Wait this long before retrying rows from a throttled partial batch.",
    )
    parser.add_argument(
        "--max-provider-cooldowns",
        type=int,
        default=0,
        help="Maximum automatic partial-batch cooldowns before stopping.",
    )
    return parser.parse_args()


class RequestStartLimiter:
    """Space concurrent request starts to avoid synchronized provider bursts."""

    def __init__(self, interval_seconds: float) -> None:
        if interval_seconds < 0:
            raise ValueError("request start interval must be non-negative")
        self.interval_seconds = interval_seconds
        self._lock = threading.Lock()
        self._next_start = 0.0

    def wait(self) -> None:
        if self.interval_seconds == 0:
            return
        with self._lock:
            now = time.perf_counter()
            delay = max(0.0, self._next_start - now)
            if delay:
                time.sleep(delay)
            self._next_start = time.perf_counter() + self.interval_seconds


class BatchScoringError(RuntimeError):
    """Carry valid sibling responses out of a partially failed paid batch."""

    def __init__(self, results: list[dict[str, Any]], failures: list[str]) -> None:
        super().__init__("OpenRouter batch failed: " + " | ".join(failures))
        self.results = results


def stable_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def request_settings(config: dict[str, Any]) -> dict[str, Any]:
    model = config["model"]
    request = config["request"]
    return {
        "endpoint": request["endpoint"],
        "model": model["openrouter_id"],
        "max_tokens": request["max_tokens"],
        "temperature": request["temperature"],
        "logprobs": True,
        "top_logprobs": request["top_logprobs"],
        "seed": request["seed"],
        "reasoning": {"effort": "none", "exclude": True},
        "provider": {
            "only": [request["provider_only"]],
            "allow_fallbacks": request["allow_fallbacks"],
            "require_parameters": request["require_parameters"],
            "data_collection": request["data_collection"],
            "enforce_distillable_text": request["enforce_distillable_text"],
            "max_price": {
                "prompt": request["provider_max_prompt_price_per_million"],
                "completion": request[
                    "provider_max_completion_price_per_million"
                ],
            },
        },
    }


def validate_config(config: dict[str, Any]) -> None:
    if config.get("status") != "frozen_evaluation":
        raise ValueError("OpenRouter Qwen benchmark config must be frozen")
    if config.get("model", {}).get("local_id") != "Qwen/Qwen3.5-27B":
        raise ValueError("local parity model must be Qwen/Qwen3.5-27B")
    if config.get("model", {}).get("openrouter_id") != (
        "qwen/qwen3.5-27b-20260224"
    ):
        raise ValueError("OpenRouter model must use the dated Qwen3.5-27B slug")
    prompt = config.get("prompt", {})
    if prompt.get("role") != "student":
        raise ValueError("intervention must use the compact student prompt")
    if prompt.get("enable_thinking") is not False:
        raise ValueError("Qwen native thinking must remain disabled")
    if prompt.get("assistant_suffix") != QWEN_NON_THINKING_ASSISTANT_SUFFIX:
        raise ValueError("Qwen non-thinking assistant boundary differs")
    if prompt.get("decision_prefix") != "Prediction:":
        raise ValueError("decision prefix must be Prediction:")
    if config.get("scoring", {}).get("tokens") != ["0", "1"]:
        raise ValueError("decision tokens must be literal 0 and 1")
    request = config.get("request", {})
    if request.get("endpoint") != "https://openrouter.ai/api/v1/chat/completions":
        raise ValueError("benchmark requires the frozen chat completions endpoint")
    if request.get("provider_only") != "Alibaba":
        raise ValueError("frozen provider must be Alibaba")
    if request.get("allow_fallbacks") is not False:
        raise ValueError("provider fallbacks must remain disabled")
    if int(request.get("max_tokens", -1)) != 8:
        raise ValueError("benchmark must use the frozen eight-token format budget")
    if int(request.get("top_logprobs", 0)) < 2:
        raise ValueError("request must return at least two top logprobs")
    if float(request.get("maximum_campaign_cost_usd", 0)) <= float(
        request.get("projected_full_cost_usd", math.inf)
    ):
        raise ValueError("campaign cost ceiling must exceed projected full cost")


def validate_parity_inputs(config: dict[str, Any]) -> list[dict[str, Any]]:
    parity = config["parity"]
    path = Path(parity["input"])
    if sha256_file(path) != parity["input_sha256"]:
        raise ValueError("full-prompt parity input checksum differs")
    rows = load_jsonl(path)
    if len(rows) != int(config["scope"]["rows"]):
        raise ValueError("full-prompt parity row count differs")
    for row in rows:
        metadata = row.get("metadata", {})
        if metadata.get("prompt_template_sha256") != parity[
            "prompt_template_sha256"
        ]:
            raise ValueError(f"full-prompt parity hash drift for id={row.get('id')!r}")
    return rows


def completion_payload(user_prompt: str, config: dict[str, Any]) -> dict[str, Any]:
    settings = request_settings(config)
    return {
        **{key: value for key, value in settings.items() if key != "endpoint"},
        "messages": [{"role": "user", "content": user_prompt}],
    }


def parse_completion_response(
    data: dict[str, Any], token_ids: list[int]
) -> tuple[str, float, float, str, dict[str, float], int]:
    try:
        choice = data["choices"][0]
        text = choice["message"]["content"]
        rows = choice["logprobs"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        preview = json.dumps(data)[:1000]
        raise RuntimeError(f"malformed OpenRouter completion: {preview}") from error
    if not isinstance(text, str) or not isinstance(rows, list):
        raise RuntimeError("completion text or token-logprob rows had the wrong type")
    match = re.fullmatch(r"\s*Prediction\s*:\s*([01])\s*", text)
    if match is None:
        raise RuntimeError(
            f"completion was not only a terminal binary prediction: {text!r}"
        )
    prediction = match.group(1)
    position = next(
        (
            index
            for index in range(len(rows) - 1, -1, -1)
            if rows[index].get("token") == prediction
        ),
        None,
    )
    if position is None:
        raise RuntimeError("terminal digit had no matching token-logprob row")
    alternatives = rows[position].get("top_logprobs")
    if not isinstance(alternatives, list):
        raise RuntimeError("completion top_logprobs row was not a list")
    top = {
        str(item["token"]): float(item["logprob"])
        for item in alternatives
        if isinstance(item, dict)
        and item.get("token") is not None
        and item.get("logprob") is not None
    }
    missing = [token for token in ("0", "1") if token not in top]
    if missing:
        raise RuntimeError(
            f"OpenRouter omitted literal logprobs {missing}; available={list(top)!r}"
        )
    if token_ids != [15, 16]:
        raise RuntimeError("pinned Qwen literal token IDs changed")
    return (
        prediction,
        float(top["0"]),
        float(top["1"]),
        text,
        top,
        position,
    )


def score_one(
    record: dict[str, Any],
    margin_prompt: str,
    prompt_tokens: int,
    token_ids: list[int],
    config: dict[str, Any],
    config_sha256: str,
    api_key: str,
    request_start_limiter: RequestStartLimiter,
) -> dict[str, Any]:
    request = config["request"]
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/gleipnir-monitoring/gleipnir",
        "X-Title": "Gleipnir Qwen prompt ablation",
        "X-OpenRouter-Metadata": "enabled",
    }
    payload = completion_payload(str(record["prompt"]), config)
    started = time.perf_counter()
    last_error: Exception | None = None
    for attempt in range(int(request["max_retries"]) + 1):
        response: requests.Response | None = None
        try:
            request_start_limiter.wait()
            response = requests.post(
                request["endpoint"],
                headers=headers,
                json=payload,
                timeout=float(request["request_timeout_seconds"]),
            )
            if response.status_code >= 400:
                raise RuntimeError(
                    f"HTTP {response.status_code} from OpenRouter: "
                    f"{response.text[:1000]}"
                )
            data = response.json()
            (
                prediction,
                logprob_0,
                logprob_1,
                generated_text,
                terminal_top_logprobs,
                label_position,
            ) = parse_completion_response(data, token_ids)
            usage = data.get("usage") or {}
            reported_prompt_tokens = int(usage.get("prompt_tokens") or 0)
            reported_completion_tokens = int(usage.get("completion_tokens") or 0)
            if not 1 <= reported_completion_tokens <= int(request["max_tokens"]):
                raise RuntimeError("OpenRouter output-token count exceeded the budget")
            provider = str(data.get("provider") or "")
            if provider.casefold() != str(request["provider_only"]).casefold():
                raise RuntimeError(f"unexpected OpenRouter provider: {provider!r}")
            metadata = record["metadata"]
            return {
                "id": str(record["id"]),
                "source": str(metadata["source_dataset"]),
                "label": int(metadata["ground_truth"]),
                "score": normalized_score(logprob_0, logprob_1),
                "prediction": int(prediction),
                "generated_token": prediction,
                "generated_text": generated_text,
                "label_position": label_position,
                "logprob_0": logprob_0,
                "logprob_1": logprob_1,
                "logit_margin_1_minus_0": logprob_1 - logprob_0,
                "terminal_top_logprobs": terminal_top_logprobs,
                "token_id_0": token_ids[0],
                "token_id_1": token_ids[1],
                "prompt_tokens": reported_prompt_tokens,
                "local_reference_prompt_tokens": prompt_tokens,
                "provider_prompt_tokens": reported_prompt_tokens,
                "provider_completion_tokens": reported_completion_tokens,
                "provider_reported_cost_usd": usage.get("cost"),
                "provider": provider,
                "model": data.get("model"),
                "response_id": data.get("id"),
                "openrouter_metadata": data.get("openrouter_metadata"),
                "source_prompt_sha256": metadata["rendered_prompt_sha256"],
                "margin_prompt_sha256": hashlib.sha256(
                    margin_prompt.encode()
                ).hexdigest(),
                "request_settings_sha256": stable_sha256(request_settings(config)),
                "config_sha256": config_sha256,
                "attempts": attempt + 1,
                "latency_seconds": time.perf_counter() - started,
            }
        except (RuntimeError, requests.RequestException, ValueError) as error:
            last_error = error
            retryable = response is None or (
                response.status_code in RETRYABLE_STATUS_CODES
            )
            if attempt >= int(request["max_retries"]) or not retryable:
                break
            time.sleep(retry_delay_seconds(attempt, response))
    raise RuntimeError(
        f"record {record['id']!r} failed after "
        f"{int(request['max_retries']) + 1} attempts: {last_error}"
    )


def validate_existing(
    rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    config: dict[str, Any],
    config_sha256: str,
) -> dict[str, dict[str, Any]]:
    allowed = {str(row["id"]): row for row in records}
    settings_sha256 = stable_sha256(request_settings(config))
    completed = {}
    for row in rows:
        record_id = str(row.get("id"))
        if record_id not in allowed:
            raise ValueError(f"cached result contains unknown id={record_id!r}")
        if row.get("config_sha256") != config_sha256:
            raise ValueError(f"cached config drift for id={record_id!r}")
        if row.get("request_settings_sha256") != settings_sha256:
            raise ValueError(f"cached request drift for id={record_id!r}")
        record = allowed[record_id]
        metadata = record["metadata"]
        if row.get("source") != metadata["source_dataset"]:
            raise ValueError(f"cached source drift for id={record_id!r}")
        if int(row.get("label", -1)) != int(metadata["ground_truth"]):
            raise ValueError(f"cached label drift for id={record_id!r}")
        if row.get("source_prompt_sha256") != metadata["rendered_prompt_sha256"]:
            raise ValueError(f"cached prompt drift for id={record_id!r}")
        if row.get("provider") != config["request"]["provider_only"]:
            raise ValueError(f"cached provider drift for id={record_id!r}")
        if re.fullmatch(
            r"\s*Prediction\s*:\s*[01]\s*", str(row.get("generated_text", ""))
        ) is None:
            raise ValueError(f"cached completion drift for id={record_id!r}")
        if not math.isfinite(float(row.get("score", math.nan))):
            raise ValueError(f"cached non-finite score for id={record_id!r}")
        if record_id in completed:
            raise ValueError(f"cached results contain duplicate id={record_id!r}")
        completed[record_id] = row
    return completed


def render_records(
    records: list[dict[str, Any]], tokenizer: Any, config: dict[str, Any]
) -> dict[str, tuple[str, int]]:
    prompt = config["prompt"]
    rendered = {}
    for row in records:
        margin_prompt = render_margin_prompt(
            tokenizer,
            str(row["prompt"]),
            enable_thinking=bool(prompt["enable_thinking"]),
            assistant_suffix=str(prompt["assistant_suffix"]),
            decision_prefix=str(prompt["decision_prefix"]),
        )
        rendered[str(row["id"])] = (
            margin_prompt,
            len(tokenizer.encode(margin_prompt, add_special_tokens=False)),
        )
    return rendered


def score_batch(
    batch: list[dict[str, Any]],
    rendered: dict[str, tuple[str, int]],
    token_ids: list[int],
    config: dict[str, Any],
    config_sha256: str,
    api_key: str,
    concurrency: int,
    request_start_limiter: RequestStartLimiter,
) -> list[dict[str, Any]]:
    results = []
    failures = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {}
        for record in batch:
            margin_prompt, prompt_tokens = rendered[str(record["id"])]
            future = pool.submit(
                score_one,
                record,
                margin_prompt,
                prompt_tokens,
                token_ids,
                config,
                config_sha256,
                api_key,
                request_start_limiter,
            )
            futures[future] = str(record["id"])
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as error:
                failures.append(f"{futures[future]}: {error}")
    if failures:
        raise BatchScoringError(results, failures)
    return results


def cost_summary(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    request = config["request"]
    prompt_tokens = sum(int(row["provider_prompt_tokens"]) for row in rows)
    completion_tokens = sum(
        int(row["provider_completion_tokens"]) for row in rows
    )
    projected = (
        prompt_tokens * float(request["provider_max_prompt_price_per_million"])
        + completion_tokens
        * float(request["provider_max_completion_price_per_million"])
    ) / 1_000_000
    reported = [row.get("provider_reported_cost_usd") for row in rows]
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "price_ceiling_projection_usd": projected,
        "provider_reported_cost_usd": (
            sum(float(value) for value in reported if value is not None)
            if any(value is not None for value in reported)
            else None
        ),
    }


def main() -> None:
    args = parse_args()
    load_dotenv(".env")
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is missing")
    config = load_json(args.config)
    validate_config(config)
    concurrency = (
        int(args.concurrency)
        if args.concurrency is not None
        else int(config["request"]["concurrency"])
    )
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    if args.max_new_rows is not None and args.max_new_rows <= 0:
        raise ValueError("max new rows must be positive")
    request_batch_rows = (
        int(args.request_batch_rows)
        if args.request_batch_rows is not None
        else int(config["request"]["request_batch_rows"])
    )
    if request_batch_rows <= 0:
        raise ValueError("request batch rows must be positive")
    if args.provider_cooldown_seconds < 0:
        raise ValueError("provider cooldown must be non-negative")
    if args.max_provider_cooldowns < 0:
        raise ValueError("maximum provider cooldowns must be non-negative")
    if args.provider_cooldown_seconds > 0 and args.max_provider_cooldowns == 0:
        raise ValueError("positive provider cooldown requires a positive maximum")
    request_start_limiter = RequestStartLimiter(
        float(args.request_start_interval_seconds)
    )
    records = validate_inputs(config)
    parity_records = validate_parity_inputs(config)
    baseline_predictions_path = Path(config["baseline"]["predictions"])
    if sha256_file(baseline_predictions_path) != config["baseline"][
        "predictions_sha256"
    ]:
        raise ValueError("local full-rubric baseline prediction checksum differs")
    config_sha256 = sha256_file(args.config)
    prompt_set = load_prompt_set()
    if prompt_set.student.template_sha256 != config["prompt"]["template_sha256"]:
        raise ValueError("working-tree compact prompt differs from frozen config")
    if prompt_set.teacher.template_sha256 != config["parity"][
        "prompt_template_sha256"
    ]:
        raise ValueError("working-tree full prompt differs from parity config")

    from transformers import AutoTokenizer

    model = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model["local_id"], revision=model["local_revision"]
    )
    token_ids = [
        tokenizer.encode(token, add_special_tokens=False)[0] for token in ("0", "1")
    ]
    if any(
        len(tokenizer.encode(token, add_special_tokens=False)) != 1
        for token in ("0", "1")
    ):
        raise ValueError("literal decision token is no longer a single token")

    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    sources = list(config["scope"]["sources"])
    parity_canary = balanced_canary_rows(
        parity_records,
        sources,
        rows_per_source_label=int(config["parity"]["rows_per_source_label"]),
    )
    parity_rendered = render_records(parity_canary, tokenizer, config)
    parity_path = args.output / "parity_predictions.jsonl"
    parity_existing = (
        [] if args.force or not parity_path.is_file() else load_jsonl(parity_path)
    )
    parity_completed = validate_existing(
        parity_existing, parity_canary, config, config_sha256
    )
    pending_parity = [
        row for row in parity_canary if str(row["id"]) not in parity_completed
    ]
    if pending_parity:
        for row in score_batch(
            pending_parity,
            parity_rendered,
            token_ids,
            config,
            config_sha256,
            api_key,
            concurrency,
            request_start_limiter,
        ):
            parity_completed[row["id"]] = row
        atomic_write_jsonl(
            parity_path, ordered_predictions(parity_canary, parity_completed)
        )
    parity_predictions = ordered_predictions(parity_canary, parity_completed)
    baseline = {
        str(row["id"]): row
        for row in load_jsonl(baseline_predictions_path)
    }
    local_scores = np.asarray(
        [float(baseline[str(row["id"])]["score"]) for row in parity_canary]
    )
    hosted_scores = np.asarray([float(row["score"]) for row in parity_predictions])
    pearson = float(np.corrcoef(local_scores, hosted_scores)[0, 1])
    mean_absolute_difference = float(np.mean(np.abs(local_scores - hosted_scores)))
    parity_passed = (
        len(parity_predictions) == len(parity_canary)
        and math.isfinite(pearson)
        and pearson >= float(config["parity"]["minimum_pearson"])
        and mean_absolute_difference
        <= float(config["parity"]["maximum_mean_absolute_difference"])
    )
    atomic_write_json(
        args.output / "parity_result.json",
        {
            "campaign_id": config["campaign_id"],
            "config_sha256": config_sha256,
            "passed": parity_passed,
            "rows": len(parity_predictions),
            "pearson": pearson,
            "mean_absolute_difference": mean_absolute_difference,
            "cost": cost_summary(parity_predictions, config),
        },
    )
    if not parity_passed:
        raise RuntimeError("OpenRouter full-prompt parity canary failed")
    print(
        f"parity complete rows={len(parity_predictions)} pearson={pearson:.6f} "
        f"mad={mean_absolute_difference:.6f}",
        flush=True,
    )

    rendered = render_records(records, tokenizer, config)
    predictions_path = args.output / "predictions.jsonl"
    existing = (
        []
        if args.force or not predictions_path.is_file()
        else load_jsonl(predictions_path)
    )
    completed = validate_existing(existing, records, config, config_sha256)
    compact_canary = balanced_canary_rows(
        records,
        sources,
        rows_per_source_label=int(config["parity"]["rows_per_source_label"]),
    )
    pending_canary = [
        row for row in compact_canary if str(row["id"]) not in completed
    ]
    if pending_canary:
        for row in score_batch(
            pending_canary,
            rendered,
            token_ids,
            config,
            config_sha256,
            api_key,
            concurrency,
            request_start_limiter,
        ):
            completed[row["id"]] = row
        atomic_write_jsonl(predictions_path, ordered_predictions(records, completed))
    canary_predictions = [completed[str(row["id"])] for row in compact_canary]
    atomic_write_json(
        args.output / "canary_result.json",
        {
            "campaign_id": config["campaign_id"],
            "config_sha256": config_sha256,
            "purpose": "compact_prompt_one_token_interface",
            "cost": cost_summary(canary_predictions, config),
            **summarize(canary_predictions),
        },
    )
    print(f"compact canary complete rows={len(canary_predictions)}", flush=True)
    if args.stop_after_canary:
        return

    pending = [row for row in records if str(row["id"]) not in completed]
    if args.max_new_rows is not None:
        pending = pending[: args.max_new_rows]
    cooldowns_used = 0
    for batch_index, batch in enumerate(batches(pending, request_batch_rows), start=1):
        remaining_batch = batch
        while remaining_batch:
            try:
                new_rows = score_batch(
                    remaining_batch,
                    rendered,
                    token_ids,
                    config,
                    config_sha256,
                    api_key,
                    concurrency,
                    request_start_limiter,
                )
            except BatchScoringError as error:
                for row in error.results:
                    completed[row["id"]] = row
                atomic_write_jsonl(
                    predictions_path, ordered_predictions(records, completed)
                )
                remaining_batch = [
                    row
                    for row in remaining_batch
                    if str(row["id"]) not in completed
                ]
                print(
                    f"partial batch={batch_index} durable={len(completed)}/"
                    f"{len(records)} failures={len(remaining_batch)}",
                    flush=True,
                )
                if (
                    args.provider_cooldown_seconds == 0
                    or cooldowns_used >= args.max_provider_cooldowns
                ):
                    raise
                cooldowns_used += 1
                print(
                    f"provider cooldown={cooldowns_used}/"
                    f"{args.max_provider_cooldowns} seconds="
                    f"{args.provider_cooldown_seconds:.1f}",
                    flush=True,
                )
                time.sleep(args.provider_cooldown_seconds)
                continue
            for row in new_rows:
                completed[row["id"]] = row
            remaining_batch = []
        predictions = ordered_predictions(records, completed)
        costs = cost_summary(predictions, config)
        parity_cost = cost_summary(parity_predictions, config)
        total_cost_ceiling = (
            costs["price_ceiling_projection_usd"]
            + parity_cost["price_ceiling_projection_usd"]
        )
        if total_cost_ceiling > float(
            config["request"]["maximum_campaign_cost_usd"]
        ):
            raise RuntimeError("campaign cost ceiling exceeded")
        atomic_write_jsonl(predictions_path, predictions)
        print(
            f"batch={batch_index} complete={len(completed)}/{len(records)} "
            f"cost_ceiling_usd={total_cost_ceiling:.4f}",
            flush=True,
        )
    if args.max_new_rows is not None and len(completed) != len(records):
        print(
            f"bounded resume complete={len(completed)}/{len(records)}",
            flush=True,
        )
        return
    predictions = ordered_predictions(records, completed)
    if len(predictions) != len(records):
        raise RuntimeError("full compact-prompt OOD benchmark is incomplete")
    costs = cost_summary(predictions, config)
    result = {
        "campaign_id": config["campaign_id"],
        "config_sha256": config_sha256,
        "model": config["model"],
        "prompt": config["prompt"],
        "request_settings": request_settings(config),
        "providers": dict(Counter(str(row["provider"]) for row in predictions)),
        "models": dict(Counter(str(row["model"]) for row in predictions)),
        "cost": costs,
        "parity_cost": cost_summary(parity_predictions, config),
        "runtime": {
            "elapsed_seconds_this_invocation": time.time() - started,
            "configured_concurrency": int(config["request"]["concurrency"]),
            "effective_concurrency": concurrency,
            "effective_request_batch_rows": request_batch_rows,
            "request_start_interval_seconds": float(
                args.request_start_interval_seconds
            ),
            "provider_cooldown_seconds": float(args.provider_cooldown_seconds),
            "provider_cooldowns_used": cooldowns_used,
        },
        "score": "normalized terminal literal 1 versus 0 OpenRouter logprob",
        **summarize(predictions),
    }
    atomic_write_json(args.output / "result.json", result)
    macro = result["metrics"]["macro"]["macro"]
    print(
        f"complete rows={len(predictions)} macro_auroc={macro['auroc']:.6f} "
        f"macro_pauroc_at_20={macro['pauroc_at_20']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
