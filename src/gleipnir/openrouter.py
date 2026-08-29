"""OpenRouter client for resumable literal binary-logprob annotation."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

import numpy as np
import requests

DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
RETRYABLE_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504})


class ResponseLike(Protocol):
    status_code: int
    text: str
    headers: Any

    def json(self) -> dict[str, Any]: ...


class OpenRouterError(RuntimeError):
    """An OpenRouter request or response could not satisfy the score contract."""


@dataclass(frozen=True)
class PromptRecord:
    """One stable prompt identity to annotate."""

    record_id: str
    prompt: str
    metadata: dict[str, Any]

    @property
    def prompt_sha256(self) -> str:
        return hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class OpenRouterConfig:
    """Reproducible request settings for binary label scoring."""

    model: str
    endpoint: str = DEFAULT_ENDPOINT
    max_tokens: int = 8
    top_logprobs: int = 5
    reasoning_effort: str = "none"
    provider_sort: str | None = "price"
    provider_order: tuple[str, ...] = ()
    provider_only: str | None = None
    allow_fallbacks: bool = True
    enforce_distillable_text: bool = True
    session_id: str | None = None
    cache_prefix: str = ""
    request_timeout: float = 180.0
    max_retries: int = 6
    app_url: str = "https://github.com/gleipnir-monitoring/gleipnir"
    app_title: str = "Gleipnir monitor distillation"


def sha256_json(value: Any) -> str:
    """Hash a JSON value using a stable, compact encoding."""
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def logsumexp(values: list[float]) -> float:
    """Stable log-sum-exp over a small list."""
    if not values:
        return -math.inf
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def binary_score_from_top_logprobs(
    top_logprobs: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    """Normalize exact literal ``0`` and ``1`` token probabilities."""
    by_label: dict[str, list[float]] = {"0": [], "1": []}
    for token, value in top_logprobs.items():
        if token not in by_label:
            continue
        if isinstance(value, dict):
            value = value.get("logprob")
        if value is not None:
            by_label[token].append(float(value))
    missing = [label for label, values in by_label.items() if not values]
    if missing:
        raise OpenRouterError(
            f"top logprobs omitted literal token(s) {missing}; "
            f"available={list(top_logprobs)[:30]!r}"
        )
    label_logprobs = {label: logsumexp(values) for label, values in by_label.items()}
    denominator = np.logaddexp(label_logprobs["0"], label_logprobs["1"])
    score = math.exp(label_logprobs["1"] - float(denominator))
    return score, label_logprobs


def extract_terminal_binary_top_logprobs(
    response_data: dict[str, Any],
) -> tuple[dict[str, float], str, int]:
    """Extract alternatives for a terminal ``Prediction:0|1`` label token."""
    import re

    try:
        choice = response_data["choices"][0]
        text = choice["message"]["content"]
        rows = choice["logprobs"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        preview = json.dumps(response_data)[:1000]
        raise OpenRouterError(
            f"response did not contain chat token logprobs: {preview}"
        ) from error
    if not isinstance(text, str) or not isinstance(rows, list):
        raise OpenRouterError("response text or token logprobs had the wrong type")

    match = re.search(r"(?i)Prediction\s*:\s*<?([01])>?\s*$", text)
    if match is None:
        raise OpenRouterError(
            f"completion lacked a terminal binary prediction: {text!r}"
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
        raise OpenRouterError(
            f"terminal prediction {prediction!r} had no matching token row"
        )
    alternatives = rows[position].get("top_logprobs")
    if not isinstance(alternatives, list) or not alternatives:
        raise OpenRouterError("terminal label top_logprobs was empty")
    top = {
        item["token"]: float(item["logprob"])
        for item in alternatives
        if isinstance(item, dict)
        and isinstance(item.get("token"), str)
        and item.get("logprob") is not None
    }
    if not top:
        raise OpenRouterError("terminal label top_logprobs had no usable entries")
    return top, text, position


def _request_body_settings(config: OpenRouterConfig) -> dict[str, Any]:
    """Return prompt-independent fields sent in the OpenRouter request body."""
    if not 1 <= config.top_logprobs <= 5:
        raise ValueError("top_logprobs must be between 1 and 5")
    if config.max_tokens < 3:
        raise ValueError("max_tokens must allow a terminal Prediction:<label>")
    if config.provider_only and config.provider_order:
        raise ValueError("provider_only and provider_order are mutually exclusive")
    if config.session_id is not None and not 1 <= len(config.session_id) <= 256:
        raise ValueError("session_id must contain between 1 and 256 characters")
    provider: dict[str, Any] = {
        "require_parameters": True,
        "data_collection": "deny",
        "allow_fallbacks": config.allow_fallbacks,
        "enforce_distillable_text": config.enforce_distillable_text,
    }
    if config.provider_order:
        provider["order"] = list(config.provider_order)
    elif config.provider_sort:
        provider["sort"] = config.provider_sort
    if config.provider_only:
        provider["only"] = [config.provider_only]
    settings: dict[str, Any] = {
        "model": config.model,
        "max_tokens": config.max_tokens,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": config.top_logprobs,
        "reasoning": {"effort": config.reasoning_effort, "exclude": True},
        "provider": provider,
    }
    if config.session_id:
        settings["session_id"] = config.session_id
    return settings


def request_settings(config: OpenRouterConfig) -> dict[str, Any]:
    """Return auditable settings without storing the potentially sensitive prefix."""
    settings = _request_body_settings(config)
    if config.cache_prefix:
        settings["cache_control"] = {
            "type": "ephemeral",
            "prefix_chars": len(config.cache_prefix),
            "prefix_sha256": hashlib.sha256(
                config.cache_prefix.encode("utf-8")
            ).hexdigest(),
        }
    return settings


def request_settings_sha256(config: OpenRouterConfig) -> str:
    """Identify all settings that can change a cached teacher annotation."""
    return sha256_json(
        {
            "endpoint": config.endpoint,
            "settings": request_settings(config),
        }
    )


def request_payload(record: PromptRecord, config: OpenRouterConfig) -> dict[str, Any]:
    """Build a privacy-conscious deterministic chat-completions payload."""
    content: str | list[dict[str, Any]] = record.prompt
    if config.cache_prefix:
        if not record.prompt.startswith(config.cache_prefix):
            raise ValueError(
                f"record {record.record_id!r} does not start with cache_prefix"
            )
        content = [
            {
                "type": "text",
                "text": config.cache_prefix,
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": record.prompt[len(config.cache_prefix) :],
            },
        ]
    return {
        **_request_body_settings(config),
        "messages": [{"role": "user", "content": content}],
    }


def cache_usage_from_response(response_data: dict[str, Any]) -> dict[str, Any]:
    """Normalize provider-cache telemetry while preserving the raw usage object."""
    usage = response_data.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    details = usage.get("prompt_tokens_details")
    if not isinstance(details, dict):
        details = {}
    cache_discount = response_data.get("cache_discount")
    if cache_discount is None:
        cache_discount = usage.get("cache_discount")
    return {
        "cached_tokens": details.get("cached_tokens", 0),
        "cache_write_tokens": details.get("cache_write_tokens", 0),
        "cache_discount": cache_discount,
    }


def retry_delay_seconds(attempt: int, response: ResponseLike | None) -> float:
    """Honor Retry-After, otherwise use capped exponential backoff."""
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return min(60.0, max(0.0, float(retry_after)))
            except ValueError:
                pass
    return min(30.0, 1.5 * (2**attempt))


def score_prompt(
    record: PromptRecord,
    config: OpenRouterConfig,
    api_key: str,
    *,
    session: requests.Session | None = None,
    sleep: Any = time.sleep,
) -> dict[str, Any]:
    """Fetch one prompt's binary logprobs with bounded retries."""
    requester = session or requests
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": config.app_url,
        "X-Title": config.app_title,
        "X-OpenRouter-Metadata": "enabled",
    }
    payload = request_payload(record, config)
    settings = request_settings(config)
    settings_sha256 = request_settings_sha256(config)
    requested_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    last_error: Exception | None = None
    for attempt in range(config.max_retries + 1):
        response: ResponseLike | None = None
        try:
            response = requester.post(
                config.endpoint,
                headers=headers,
                json=payload,
                timeout=config.request_timeout,
            )
            if response.status_code >= 400:
                raise OpenRouterError(
                    f"HTTP {response.status_code} from OpenRouter: "
                    f"{response.text[:1000]}"
                )
            data = response.json()
            top, generated_text, label_position = extract_terminal_binary_top_logprobs(
                data
            )
            score, label_logprobs = binary_score_from_top_logprobs(top)
            choice = data["choices"][0]
            usage = data.get("usage") or {}
            return {
                "id": record.record_id,
                "prompt_sha256": record.prompt_sha256,
                "prompt_chars": len(record.prompt),
                "metadata": record.metadata,
                "request_settings": settings,
                "request_settings_sha256": settings_sha256,
                "requested_at": requested_at,
                "received_at": datetime.now(UTC).isoformat(),
                "model": data.get("model"),
                "provider": data.get("provider"),
                "response_id": data.get("id"),
                "created": data.get("created"),
                "text": generated_text,
                "finish_reason": choice.get("finish_reason"),
                "label_position": label_position,
                "score": score,
                "label_logprobs": label_logprobs,
                "target_logprobs": {
                    "negative": label_logprobs["0"],
                    "positive": label_logprobs["1"],
                },
                "target_probs": {"negative": 1.0 - score, "positive": score},
                "top_logprobs": top,
                "usage": usage,
                "cache_usage": cache_usage_from_response(data),
                "openrouter_metadata": data.get("openrouter_metadata"),
                "latency_seconds": time.perf_counter() - started,
                "attempts": attempt + 1,
            }
        except (OpenRouterError, requests.RequestException, ValueError) as error:
            last_error = error
            retryable = (
                response is None or response.status_code in RETRYABLE_STATUS_CODES
            )
            if attempt >= config.max_retries or not retryable:
                break
            sleep(retry_delay_seconds(attempt, response))
    raise OpenRouterError(
        f"record {record.record_id!r} failed after "
        f"{config.max_retries + 1} attempts: {last_error}"
    )
