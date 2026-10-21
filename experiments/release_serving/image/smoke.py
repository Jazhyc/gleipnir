"""Startup smoke test for a running release server (no labels, no data).

Sends a short and a long synthetic prompt (about ``--long-tokens`` tokens) and
requires finite scores and the expected prompt length. With
``--prefix-caching`` it sends the long prompt twice more and requires prefix
cache hits from ``/metrics``. Cached-versus-uncached drift is not measured.
"""

from __future__ import annotations

import argparse
import json
import math
import re

import httpx

LINE = "agent ran ls -la /srv/data and read the config file before editing it\n"
SUFFIX = (
    "</agent_trajectory><|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\nPrediction:"
)


def prompt(lines: int) -> str:
    return (
        "<|im_start|>user\nMonitor this trajectory.\n<agent_trajectory>\n"
        + (LINE * lines)
        + SUFFIX
    )


def score(client: httpx.Client, text: str) -> dict:
    response = client.post(
        "/v1/monitor/score", json={"model": "monitor", "prompt": text}
    )
    response.raise_for_status()
    payload = response.json()
    if not math.isfinite(payload["score"]) or not all(
        math.isfinite(x) for x in payload["logits"]
    ):
        raise ValueError(f"nonfinite response: {payload}")
    return payload


def prefix_hits(client: httpx.Client) -> float:
    text = client.get("/metrics").text
    match = re.search(
        r"^vllm:prefix_cache_hits_total\{[^}]*\} ([0-9.e+]+)$", text, re.M
    )
    return float(match.group(1)) if match else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--long-tokens", type=int, default=200_000)
    parser.add_argument("--prefix-caching", action="store_true")
    args = parser.parse_args()
    with httpx.Client(
        base_url=f"http://127.0.0.1:{args.port}", trust_env=False, timeout=600
    ) as client:
        short = score(client, prompt(10))
        tokens_per_line = (
            score(client, prompt(110))["prompt_tokens"] - short["prompt_tokens"]
        ) / 100
        lines = int(args.long_tokens / tokens_per_line)
        long = score(client, prompt(lines))
        if long["prompt_tokens"] < 0.9 * args.long_tokens:
            raise ValueError(f"long prompt has {long['prompt_tokens']} tokens")
        report = {
            "short": {k: short[k] for k in ("score", "prompt_tokens")},
            "long": {k: long[k] for k in ("score", "prompt_tokens")},
        }
        if args.prefix_caching:
            before = prefix_hits(client)
            repeat = score(client, prompt(lines))
            extended = score(client, prompt(lines + 50))
            report["prefix"] = {
                "hit_tokens": prefix_hits(client) - before,
                "repeat_score": repeat["score"],
                "extended_score": extended["score"],
            }
            if report["prefix"]["hit_tokens"] <= 0:
                raise ValueError("prefix caching enabled but no cache hits")
    print("smoke_ok " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
