"""Attacker-model calls through OpenRouter, with a hard dollar cap.

Used by attacks that need free text from an LLM (A2 cover comments, A4
rewrites). Keys come from the environment and are never stored or logged.
Every call reserves its worst-case cost before it is sent.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any

URL = "https://openrouter.ai/api/v1/chat/completions"


class AttackerError(RuntimeError):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}


def error_details(error: Exception, key: str) -> dict:
    """Keep actionable metadata, never request headers or raw provider bodies."""
    details = {"exception_type": type(error).__name__}
    if isinstance(error, AttackerError):
        details.update(error.details)
    if isinstance(error, urllib.error.HTTPError):
        details["http_status"] = error.code
        for name in ("retry-after", "x-request-id", "x-openrouter-request-id"):
            if error.headers and error.headers.get(name):
                details[name] = error.headers[name][:160]
        try:
            body = json.loads(error.read(16384))
            info = body.get("error", {})
            if isinstance(info, dict):
                for name in ("code", "message"):
                    if name in info:
                        details["provider_" + name] = str(info[name])[:1000]
        except (ValueError, OSError, AttributeError):
            details["body_parse_failed"] = True
    for name, value in list(details.items()):
        if isinstance(value, str):
            value = value.replace(key, "[REDACTED]") if key else value
            value = re.sub(r"(?i)bearer\s+\S+|sk-or-[\w-]+", "[REDACTED]", value)
            details[name] = value[:500]
    return details


class BudgetExceeded(AttackerError):
    pass


@dataclass(frozen=True)
class AttackerConfig:
    model: str
    providers: tuple[str, ...]
    temperature: float
    max_tokens: int
    reasoning_effort: str
    usd_per_m_prompt: float
    usd_per_m_completion: float

    def payload(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "reasoning": {"effort": self.reasoning_effort, "exclude": True},
            "provider": {
                "order": list(self.providers),
                "allow_fallbacks": False,
                "require_parameters": True,
            },
            "usage": {"include": True},
        }

    def worst_case_usd(self, prompt_chars: int) -> float:
        # Caller supplies UTF-8 bytes; add an allowance for chat framing.
        return (
            (prompt_chars + 2048) * self.usd_per_m_prompt
            + self.max_tokens * self.usd_per_m_completion
        ) / 1e6


@dataclass
class Budget:
    """Thread-safe: concurrent calls count their outstanding worst case."""

    cap_usd: float
    spent_usd: float = 0.0
    calls: int = 0
    outstanding_usd: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def reserve(self, usd: float) -> None:
        with self._lock:
            if self.spent_usd + self.outstanding_usd + usd > self.cap_usd:
                raise BudgetExceeded(
                    f"spent ${self.spent_usd:.4f}, outstanding "
                    f"${self.outstanding_usd:.4f}, next ${usd:.4f}, "
                    f"cap ${self.cap_usd:.2f}"
                )
            self.outstanding_usd += usd

    def settle(self, reserved: float, usd: float) -> None:
        with self._lock:
            self.outstanding_usd -= reserved
            self.spent_usd += usd
            self.calls += 1


def _post(body: dict[str, Any], key: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        URL,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def complete(
    messages: list[dict[str, str]],
    config: AttackerConfig,
    budget: Budget,
    *,
    key: str | None = None,
    retries: int = 1,
    timeout: float = 300.0,
    post: Any = _post,
) -> dict[str, Any]:
    """One chat completion; returns text, finish reason, usage, cost, provider."""
    key = key or os.environ["OPENROUTER_API_KEY"]
    if retries != 1:
        raise ValueError("Paid calls must not retry ambiguous failures")
    body = config.payload(messages)
    reserved = config.worst_case_usd(
        sum(len(m["content"].encode("utf-8")) for m in messages)
    )
    budget.reserve(reserved)
    try:
        data = post(body, key, timeout)
        if "error" in data:
            info = data["error"]
            details = {"response_error": True}
            if isinstance(info, dict):
                details.update(
                    {
                        "provider_" + k: str(info[k])[:1000]
                        for k in ("code", "message")
                        if k in info
                    }
                )
            raise AttackerError("provider returned an error response", details)
        choice = data["choices"][0]
        usage = data.get("usage") or {}
        cost = float(usage["cost"]) if usage.get("cost") is not None else reserved
        if not math.isfinite(cost) or cost < 0:
            raise AttackerError("invalid provider cost")
        result = {
            "text": choice["message"].get("content") or "",
            "finish_reason": choice.get("finish_reason"),
            "usage": {
                k: usage.get(k)
                for k in ("prompt_tokens", "completion_tokens", "total_tokens")
            },
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get(
                "reasoning_tokens"
            ),
            "cost_usd": cost,
            "provider": data.get("provider"),
            "model": data.get("model"),
            "response_id": data.get("id"),
        }
    except Exception as error:
        budget.settle(reserved, reserved)
        details = error_details(error, key)
        status = details.get("http_status", details.get("provider_code", "unknown"))
        raise AttackerError(
            f"attacker call failed: {type(error).__name__}; status={status}", details
        ) from error
    budget.settle(reserved, cost)
    return result


def describe(config: AttackerConfig) -> dict[str, Any]:
    return asdict(config)
