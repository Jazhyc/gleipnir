"""Direct OpenAI Responses parsing and audited binary monitoring requests."""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from gleipnir.openrouter import RETRYABLE_STATUS_CODES, retry_delay_seconds


class TokenStartLimiter:
    """Pace request starts under a shared token rate with forty worker slots."""

    def __init__(self, tokens_per_minute: int) -> None:
        if tokens_per_minute <= 0:
            raise ValueError("token rate must be positive")
        self.rate = tokens_per_minute / 60
        self.lock = threading.Lock()
        self.next_start = 0.0

    def wait(self, tokens: float) -> None:
        """Reserve one start time; count every HTTP attempt toward the limit."""
        with self.lock:
            now = time.perf_counter()
            start = max(now, self.next_start)
            self.next_start = start + tokens / self.rate
        time.sleep(max(0.0, start - time.perf_counter()))


def digest(value: Any) -> str:
    """Return a stable JSON identity."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def parse_response(raw: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    """Require the exact prediction line and both literal terminal logprobs."""
    if raw.get("status") != "completed" or raw.get("model") != settings["model"]:
        raise ValueError("response did not complete with the requested model")
    for field in ("temperature", "top_p", "service_tier"):
        if raw.get(field) != settings[field]:
            raise ValueError(f"effective request setting differs: {field}")
    if raw.get("reasoning", {}).get("effort") != settings["reasoning"]["effort"]:
        raise ValueError("effective reasoning effort differs")
    texts, tokens = [], []
    for item in raw.get("output", []):
        if item.get("type") != "message":
            raise ValueError("unexpected non-message response output")
        for content in item.get("content", []):
            if content.get("type") != "output_text":
                raise ValueError("response included a refusal or non-text content")
            texts.append(content["text"])
            tokens.extend(content.get("logprobs") or [])
    text = "".join(texts)
    match = re.fullmatch(r"\s*Prediction:([01])\s*", text)
    if match is None or "".join(t["token"] for t in tokens) != text:
        raise ValueError("completion or token text violated the prediction contract")
    prediction = match.group(1)
    label_character = text.index(prediction, text.index(":"))
    offset, position = 0, None
    for index, token in enumerate(tokens):
        if offset == label_character and token["token"] == prediction:
            position = index
        offset += len(token["token"])
    if position is None:
        raise ValueError("terminal label was not an exact literal digit token")
    terminal = tokens[position]
    pair = {
        item["token"]: float(item["logprob"])
        for item in [terminal, *(terminal.get("top_logprobs") or [])]
        if item.get("token") in {"0", "1"}
    }
    if set(pair) != {"0", "1"}:
        raise ValueError("terminal top-logprobs omitted a literal decision token")
    # Preserve tiny positive server roundoff; do not clamp or synthesize scores.
    if any(not math.isfinite(v) or v <= -9999 or v > 1e-4 for v in pair.values()):
        raise ValueError("literal logprobs exceed the frozen numerical tolerance")
    usage = raw["usage"]
    if usage.get("output_tokens_details", {}).get("reasoning_tokens") != 0:
        raise ValueError("response reported reasoning tokens or omitted their count")
    weights = {k: math.exp(v - max(pair.values())) for k, v in pair.items()}
    return {
        "text": text,
        "generated_label": int(prediction),
        "terminal_token_position": position,
        "label_logprobs": pair,
        "score": weights["1"] / sum(weights.values()),
        "logprob_margin": pair["1"] - pair["0"],
        "positive_logprob_roundoff": any(v > 0 for v in pair.values()),
    }


def conservative_cost(usage: dict[str, Any]) -> float:
    """Price every input token at the cache-write rate as a budget bound."""
    return (
        usage.get("input_tokens", 0) * 0.125 + usage.get("output_tokens", 0) * 0.50
    ) / 1e6


class AuditedClient:
    """Append every HTTP attempt, limit spending, and retry transient errors."""

    def __init__(
        self,
        key: str,
        settings: dict[str, Any],
        root: Path,
        *,
        budget_usd: float,
        max_attempts: int,
        timeout_seconds: float,
        tokens_per_minute: int | None = None,
        coverage_repeats: int = 0,
    ) -> None:
        self.key = key
        self.settings = settings
        self.root = root
        self.max_attempts = max_attempts
        self.timeout_seconds = timeout_seconds
        self.coverage_repeats = coverage_repeats
        self.budget_usd = budget_usd
        self.lock = threading.Lock()
        self.local = threading.local()
        self.start_limiter = (
            TokenStartLimiter(tokens_per_minute) if tokens_per_minute else None
        )
        self.reserved = 0.0
        self.spent_bound = 0.0
        self.attempt_path = root / "attempts.jsonl"
        if self.attempt_path.exists():
            with self.attempt_path.open() as handle:
                for line in handle:
                    record = json.loads(line)
                    self.spent_bound += record["conservative_cost_usd"]

    def score(self, row: dict[str, Any]) -> dict[str, Any]:
        """Score one immutable prompt, recording failures before raising."""
        body = dict(self.settings, input=[{"role": "user", "content": row["prompt"]}])
        request_hash = digest(body)
        token_proxy = row["metadata"].get(
            "kimi_raw_prompt_tokens", len(row["prompt"].encode())
        )
        estimate = (
            (token_proxy * 1.30 + 32) * 0.125
            + self.settings["max_output_tokens"] * 0.50
        ) / 1e6
        if not hasattr(self.local, "session"):
            self.local.session = requests.Session()
        coverage_repeats = 0
        for attempt in range(1, self.max_attempts + 1):
            if self.start_limiter:
                self.start_limiter.wait(token_proxy * 1.05 + 32 + 16)
            with self.lock:
                if self.spent_bound + self.reserved + estimate > self.budget_usd:
                    raise RuntimeError("campaign conservative budget bound exhausted")
                self.reserved += estimate
            started = time.perf_counter()
            record: dict[str, Any] = {
                "id": row["id"],
                "attempt": attempt,
                "started_at_utc": datetime.now(UTC).isoformat(),
                "request_sha256": request_hash,
                "prompt_sha256": hashlib.sha256(row["prompt"].encode()).hexdigest(),
                "request_settings_sha256": digest(self.settings),
                "coverage_repeat_index": coverage_repeats,
            }
            response = None
            retry = False
            try:
                response = self.local.session.post(
                    "https://api.openai.com/v1/responses",
                    json=body,
                    headers={"Authorization": f"Bearer {self.key}"},
                    timeout=(30, self.timeout_seconds),
                )
                record["http_status"] = response.status_code
                record["request_id"] = response.headers.get("x-request-id")
                record["rate_limit_headers"] = {
                    k: v
                    for k, v in response.headers.items()
                    if k.lower().startswith("x-ratelimit") or k.lower() == "retry-after"
                }
                try:
                    raw = response.json()
                except ValueError:
                    raw = {"non_json_body": response.text[:1000]}
                record["raw_response"] = raw
                record["conservative_cost_usd"] = conservative_cost(
                    raw.get("usage") or {}
                )
                retry = response.status_code in RETRYABLE_STATUS_CODES
                if response.status_code == 200:
                    try:
                        record["parsed"] = parse_response(raw, self.settings)
                    except (KeyError, TypeError, ValueError) as error:
                        record["parse_failure"] = str(error)
                        if (
                            str(error)
                            == "terminal top-logprobs omitted a literal decision token"
                            and coverage_repeats < self.coverage_repeats
                        ):
                            coverage_repeats += 1
                            retry = True
                            record["coverage_repeat_planned"] = coverage_repeats
            except requests.RequestException as error:
                record["transport_error"] = str(error).replace(self.key, "[REDACTED]")
                # A timeout can hide a billed response: retain the reservation.
                record["conservative_cost_usd"] = estimate
                retry = True
            record["latency_seconds"] = time.perf_counter() - started
            record["completed_at_utc"] = datetime.now(UTC).isoformat()
            with self.lock:
                with self.attempt_path.open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                    handle.flush()
                self.reserved -= estimate
                self.spent_bound += record["conservative_cost_usd"]
            if "parsed" in record:
                raw = record["raw_response"]
                usage = raw["usage"]
                return {
                    "id": row["id"],
                    "prompt_sha256": record["prompt_sha256"],
                    "request_sha256": request_hash,
                    "request_settings_sha256": digest(self.settings),
                    "request_settings": self.settings,
                    "timestamp_utc": record["completed_at_utc"],
                    "model": raw["model"],
                    "provider": "OpenAI",
                    "service_tier": raw["service_tier"],
                    "request_id": record["request_id"],
                    "response_id": raw["id"],
                    "coverage_repeats": coverage_repeats,
                    "latency_seconds": record["latency_seconds"],
                    "usage": {
                        "prompt_tokens": usage["input_tokens"],
                        "completion_tokens": usage["output_tokens"],
                        "reasoning_tokens": usage["output_tokens_details"][
                            "reasoning_tokens"
                        ],
                    },
                    "cache_usage": {
                        "cached_tokens": usage["input_tokens_details"]["cached_tokens"],
                        "cache_write_tokens": usage["input_tokens_details"].get(
                            "cache_write_tokens"
                        ),
                    },
                    "raw_usage": usage,
                    **record["parsed"],
                }
            if not retry or attempt == self.max_attempts:
                status = record.get("http_status", "transport")
                reason = record.get("parse_failure", "request failed")
                raise RuntimeError(f"id={row['id']} status={status}: {reason}")
            # Retry-After is a minimum; avoid forty synchronized one-second loops.
            delay = max(
                retry_delay_seconds(attempt, response),
                retry_delay_seconds(attempt, None),
            )
            if response is not None:
                try:
                    delay = max(delay, float(response.headers.get("Retry-After", 0)))
                except ValueError:
                    pass
            time.sleep(delay)
        raise AssertionError("unreachable")
