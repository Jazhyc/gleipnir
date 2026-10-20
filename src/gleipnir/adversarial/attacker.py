"""Attacker-model calls through OpenRouter, with a hard dollar cap.

Used by attacks that need free text from an LLM (A2 cover comments, A4
rewrites). Keys come from the environment and are never stored or logged.
Every call reserves its worst-case cost before it is sent.
"""

from __future__ import annotations

import json
import math
import os
import threading
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any

URL = "https://openrouter.ai/api/v1/chat/completions"


class AttackerError(RuntimeError):
    pass


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
            raise AttackerError("provider returned an error response")
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
        raise AttackerError(f"attacker call failed: {type(error).__name__}") from error
    budget.settle(reserved, cost)
    return result


def describe(config: AttackerConfig) -> dict[str, Any]:
    return asdict(config)
