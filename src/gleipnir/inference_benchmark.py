"""Frozen workload selection and strict binary HTTP inference measurements."""

from __future__ import annotations

import math
import random
from typing import Any


def match_selection(rows: list[dict], selection: list[dict]) -> list[dict]:
    """Recover the exact ordered cohort, rejecting duplicate or drifted identities."""

    def key(row: dict) -> tuple[str, str]:
        return str(row["dataset"]), str(row["index"])

    by_key = {key(row): row for row in rows}
    if len(by_key) != len(rows) or len({key(row) for row in selection}) != len(
        selection
    ):
        raise ValueError("duplicate workload identity")
    matched = []
    for item in selection:
        row = by_key[key(item)]
        if row["label"] != item["label"]:
            raise ValueError("selection drift: label")
        # Legacy rows omit this field in both source and selection. The complete
        # source-file checksum still binds their data.
        if row.get("trajectory_sha256") != item.get("trajectory_sha256"):
            raise ValueError("selection drift: trajectory_sha256")
        matched.append(row)
    return matched


def quick_workload(rows: list[dict], count: int, seed: int) -> list[dict]:
    """Select evenly spaced length ranks, including both extremes, then shuffle."""
    if not 2 <= count <= len(rows):
        raise ValueError("workload count outside parent population")
    ordered = sorted(rows, key=lambda r: (r["prompt_tokens"], r["id"]))
    positions = [round(i * (len(rows) - 1) / (count - 1)) for i in range(count)]
    selected = [ordered[i] for i in positions]
    random.Random(seed).shuffle(selected)
    return selected


def response_score(response: dict, ids: list[int], prompt_tokens: int) -> dict:
    """Require a complete one-token response and both finite decision logprobs."""
    usage = response["usage"]
    if usage["prompt_tokens"] != prompt_tokens or usage["completion_tokens"] != 1:
        raise ValueError("prompt truncation or response length mismatch")
    (choice,) = response["choices"]
    logprobs = choice["logprobs"]["top_logprobs"]
    if len(logprobs) != 1:
        raise ValueError("missing one-token logprobs")
    values = [float(logprobs[0][f"token_id:{token}"]) for token in ids]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("nonfinite decision logprob")
    margin = values[1] - values[0]
    return {
        "score": 1.0 / (1.0 + math.exp(-max(-80.0, min(80.0, margin)))),
        "logprob_0": values[0],
        "logprob_1": values[1],
        "margin": margin,
    }


def percentile(values: list[float], quantile: float) -> float:
    """Compute a linearly interpolated sample percentile."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def measurement_summary(rows: list[dict], seconds: float) -> dict[str, Any]:
    """Summarize closed-loop throughput and observed request latency by length."""
    if not rows or seconds <= 0:
        raise ValueError("empty or invalid measurement")

    def latencies(population: list[dict]) -> dict:
        values = [r["latency_seconds"] for r in population]
        return {
            "rows": len(values),
            **{
                f"p{int(q * 100)}_seconds": percentile(values, q)
                for q in (0.5, 0.95, 0.99)
            },
        }

    bins = {}
    for name, lower, upper in [
        ("lt4k", 0, 4096),
        ("4k_to16k", 4096, 16384),
        ("ge16k", 16384, math.inf),
    ]:
        population = [r for r in rows if lower <= r["prompt_tokens"] < upper]
        if population:
            bins[name] = latencies(population)
    return {
        "seconds": seconds,
        "requests_per_second": len(rows) / seconds,
        "prompt_tokens_per_second": sum(r["prompt_tokens"] for r in rows) / seconds,
        "latency": latencies(rows),
        "latency_by_prompt_length": bins,
    }
