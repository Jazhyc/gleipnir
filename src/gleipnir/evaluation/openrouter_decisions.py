"""OpenRouter Decisions client for resumable single-Noul classification.

Ported from the Jev 1.13 benchmark client (commit 5c6db2a), with the provider
pin taken from the request config instead of being fixed to TypeSafe.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

import requests

DEFAULT_DECISIONS_ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
RETRYABLE_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504, 524, 529})
# Statuses that reject the input itself (for example an over-length state).
INPUT_REJECTION_STATUS_CODES = frozenset({400, 413, 422})


class ResponseLike(Protocol):
    status_code: int
    text: str
    headers: Any

    def json(self) -> dict[str, Any]: ...


class OpenRouterDecisionsError(RuntimeError):
    """A Decisions request or response did not satisfy the frozen contract."""

    def __init__(self, message: str, *, input_rejected: bool = False) -> None:
        super().__init__(message)
        self.input_rejected = input_rejected


@dataclass(frozen=True)
class DecisionRecord:
    """One stable state to classify."""

    record_id: str
    state: str
    metadata: dict[str, Any]

    @property
    def state_sha256(self) -> str:
        return hashlib.sha256(self.state.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DecisionsConfig:
    """Reproducible settings for one Noul question on one pinned provider."""

    model: str
    provider: str
    question_id: str
    question: dict[str, Any]
    endpoint: str = DEFAULT_DECISIONS_ENDPOINT
    data_collection: str = "deny"
    request_timeout: float = 180.0
    max_retries: int = 6
    app_url: str = "https://github.com/gleipnir-monitoring/gleipnir"
    app_title: str = "Gleipnir decision-model OOD benchmark"


def stable_sha256(value: Any) -> str:
    """Hash a JSON-compatible value using a stable compact encoding."""
    encoded = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_config(config: DecisionsConfig) -> None:
    if config.endpoint != DEFAULT_DECISIONS_ENDPOINT:
        raise ValueError("Decisions requests must use the frozen OpenRouter endpoint")
    if not config.model or not config.provider or not config.question_id:
        raise ValueError("model, provider and question_id must be nonempty")
    if config.question.get("type") != "noul":
        raise ValueError("the benchmark requires one Noul question")
    if not config.question.get("instructions"):
        raise ValueError("Noul instructions must be nonempty")
    criteria = config.question.get("criteria")
    if not isinstance(criteria, dict) or set(criteria) != {"true", "false"}:
        raise ValueError("Noul criteria must define exactly true and false")
    if config.data_collection != "deny":
        raise ValueError("provider data collection must be denied")
    if config.request_timeout <= 0 or config.max_retries < 0:
        raise ValueError("timeout must be positive and retries nonnegative")


def request_settings(config: DecisionsConfig) -> dict[str, Any]:
    """Return the exact state-independent request identity."""
    validate_config(config)
    return {
        "endpoint": config.endpoint,
        "model": config.model,
        "question_id": config.question_id,
        "question_sha256": stable_sha256(config.question),
        "provider": {
            "only": [config.provider],
            "allow_fallbacks": False,
            "data_collection": config.data_collection,
        },
    }


def request_settings_sha256(config: DecisionsConfig) -> str:
    return stable_sha256(request_settings(config))


def request_payload(record: DecisionRecord, config: DecisionsConfig) -> dict[str, Any]:
    settings = request_settings(config)
    return {
        "model": settings["model"],
        "state": record.state,
        "questions": {config.question_id: config.question},
        "provider": settings["provider"],
    }


def parse_noul_response(
    data: dict[str, Any], *, question_id: str
) -> tuple[float, dict[str, Any], dict[str, Any]]:
    """Extract and validate one continuous Noul answer and its usage."""
    try:
        answers = data["answers"]
        answer = answers[question_id]
        answer_type = answer["type"]
        raw_score = answer["noul"]
        usage = data["usage"]
    except (KeyError, TypeError) as error:
        preview = json.dumps(data, ensure_ascii=False)[:1000]
        raise OpenRouterDecisionsError(
            f"malformed Decisions response: {preview}"
        ) from error
    if set(answers) != {question_id}:
        raise OpenRouterDecisionsError(f"answer keys differ: {sorted(answers)!r}")
    if answer_type != "noul":
        raise OpenRouterDecisionsError(f"unexpected answer type: {answer_type!r}")
    if isinstance(raw_score, bool) or not isinstance(raw_score, int | float):
        raise OpenRouterDecisionsError("Noul score was not numeric")
    score = float(raw_score)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise OpenRouterDecisionsError(f"Noul score outside [0, 1]: {score!r}")
    if not isinstance(usage, dict):
        raise OpenRouterDecisionsError("Decisions usage was not an object")
    for key in ("input_tokens", "output_tokens"):
        value = usage.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise OpenRouterDecisionsError(f"invalid usage field {key}: {value!r}")
    return score, dict(answer), dict(usage)


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


def score_decision(
    record: DecisionRecord,
    config: DecisionsConfig,
    api_key: str,
    *,
    session: Any = None,
    sleep: Any = time.sleep,
) -> dict[str, Any]:
    """Fetch one Noul score with bounded retries and complete provenance.

    Raises ``OpenRouterDecisionsError`` with ``input_rejected=True`` when the
    provider rejects the input itself (such as an over-length state), so callers
    can record the row; every other failure is fatal to the caller.
    """
    validate_config(config)
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
    requested_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    last_error: Exception | None = None
    rejected = fatal = False
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
                raise OpenRouterDecisionsError(
                    f"HTTP {response.status_code} from OpenRouter: "
                    f"{response.text[:1000]}"
                )
            data = response.json()
            score, answer, usage = parse_noul_response(
                data, question_id=config.question_id
            )
            provider = str(data.get("provider") or "")
            if provider.casefold() != config.provider.casefold():
                raise OpenRouterDecisionsError(
                    f"unexpected OpenRouter provider: {provider!r}"
                )
            return {
                "id": record.record_id,
                "state_sha256": record.state_sha256,
                "state_chars": len(record.state),
                "metadata": record.metadata,
                "request_settings": settings,
                "request_settings_sha256": stable_sha256(settings),
                "requested_at": requested_at,
                "received_at": datetime.now(UTC).isoformat(),
                "model": data.get("model"),
                "provider": provider,
                "response_id": data.get("id"),
                "score": score,
                "answer": answer,
                "usage": usage,
                "latency_seconds": time.perf_counter() - started,
                "attempts": attempt + 1,
            }
        except (
            OpenRouterDecisionsError,
            requests.RequestException,
            ValueError,
        ) as error:
            last_error = error
            status = None if response is None else response.status_code
            rejected = status in INPUT_REJECTION_STATUS_CODES
            fatal = (
                status is not None
                and 400 <= status < 500
                and status not in RETRYABLE_STATUS_CODES
            )
            if fatal or attempt >= config.max_retries:
                break
            sleep(retry_delay_seconds(attempt, response))
    raise OpenRouterDecisionsError(
        f"record {record.record_id!r} failed after {attempt + 1} attempts: "
        f"{last_error}",
        input_rejected=rejected,
    )
