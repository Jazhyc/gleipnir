"""Chat calls to an OpenAI-compatible endpoint (Tinker), through `attacker.complete`.

`attacker.complete` already reserves each call's worst-case cost, settles the
actual cost and redacts keys from errors. Two pieces are endpoint-specific:
the request body (no OpenRouter routing fields) and the cost, which Tinker
does not return, so it is computed from the returned token counts and the
configured prices. With `dry_run`, a local responder replaces the network.
"""

from __future__ import annotations

import json
import re
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from experiments.adversarial_code_or_comments.common import load_key
from gleipnir.adversarial.attacker import AttackerConfig, Budget, complete

THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


@dataclass(frozen=True)
class EndpointConfig(AttackerConfig):
    base_url: str = ""
    key_env: str = ""
    top_p: float | None = None
    seed: int | None = None
    extra_body_json: str = "{}"

    def payload(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.top_p is not None:
            body["top_p"] = self.top_p
        if self.seed is not None:
            body["seed"] = self.seed
        body.update(json.loads(self.extra_body_json))
        return body


def endpoint(spec: dict[str, Any]) -> EndpointConfig:
    return EndpointConfig(
        model=spec["model"],
        providers=(spec["provider"],),
        temperature=spec["temperature"],
        max_tokens=spec["max_tokens"],
        reasoning_effort="see extra_body",
        usd_per_m_prompt=spec["usd_per_m_prompt"],
        usd_per_m_completion=spec["usd_per_m_completion"],
        base_url=spec["base_url"],
        key_env=spec["key_env"],
        top_p=spec.get("top_p"),
        seed=spec.get("seed"),
        extra_body_json=json.dumps(spec.get("extra_body", {}), sort_keys=True),
    )


def priced(data: dict[str, Any], config: EndpointConfig) -> dict[str, Any]:
    """Add `usage.cost` from token counts when the endpoint does not report it."""
    usage = data.get("usage") or {}
    prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if usage.get("cost") is None and prompt is not None and completion is not None:
        usage["cost"] = (
            prompt * config.usd_per_m_prompt + completion * config.usd_per_m_completion
        ) / 1e6
        data["usage"] = usage
    return data


def http_post(config: EndpointConfig) -> Callable[..., dict[str, Any]]:
    def post(body: dict[str, Any], key: str, timeout: float) -> dict[str, Any]:
        request = urllib.request.Request(
            config.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                # Tinker's Cloudflare front returns 403 to urllib's default agent.
                "User-Agent": "gleipnir-adversarial/1.0 (+python-urllib)",
            },
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return priced(json.load(response), config)

    return post


def mock_post(responder: Callable[[list[dict[str, str]]], str]):
    """A network-free stand-in: `responder(messages)` returns the reply text."""

    def post(body: dict[str, Any], key: str, timeout: float) -> dict[str, Any]:
        text = responder(body["messages"])
        chars = sum(len(m["content"]) for m in body["messages"])
        return {
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": chars // 4,
                "completion_tokens": len(text) // 4,
                "cost": 0.0,
            },
            "model": "dry-run",
            "id": "dry-run",
        }

    return post


class Client:
    """One model role (writer or checker) with its own budget."""

    def __init__(
        self,
        spec: dict[str, Any],
        budget_usd: float,
        cfg: dict[str, Any],
        *,
        timeout: float = 300.0,
        responder: Callable[[list[dict[str, str]]], str] | None = None,
    ):
        self.config = endpoint(spec)
        self.budget = Budget(budget_usd)
        self.timeout = timeout
        if responder is None:
            self.key = load_key(self.config.key_env, cfg)
            self.post = http_post(self.config)
        else:
            self.key = "dry-run"
            self.post = mock_post(responder)

    def chat(self, messages: list[dict[str, str]], attempt: int = 1) -> dict[str, Any]:
        """One call; the seed moves with the attempt so retries are new samples."""
        config = self.config
        if config.seed is not None:
            config = replace(config, seed=config.seed + attempt - 1)
        return complete(
            messages,
            config,
            self.budget,
            key=self.key,
            timeout=self.timeout,
            post=self.post,
        )


def finish_status(result: dict[str, Any]) -> str | None:
    """`truncation` or `refusal` from the finish reason, else None."""
    if result.get("finish_reason") == "length":
        return "truncation"
    if result.get("finish_reason") in ("content_filter", "refusal"):
        return "refusal"
    return None


def clean_text(text: str) -> str:
    """Drop visible reasoning blocks and an enclosing code fence."""
    return FENCE.sub("", THINK.sub("", text).strip()).strip()


def parse_json_object(text: str) -> dict[str, Any]:
    """The reply's JSON object; tolerates reasoning blocks and code fences."""
    body = clean_text(text)
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in reply")
    value = json.loads(body[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("reply is not a JSON object")
    return value
