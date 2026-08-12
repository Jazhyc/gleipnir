"""Pure helpers for classifier-throughput benchmarks."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_length_summary(lengths: list[int]) -> dict[str, float | int]:
    if not lengths:
        raise ValueError("token lengths must not be empty")
    ordered = sorted(lengths)

    def percentile(fraction: float) -> int:
        index = math.ceil(fraction * len(ordered)) - 1
        return ordered[max(0, index)]

    return {
        "total": sum(ordered),
        "mean": statistics.fmean(ordered),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "max": ordered[-1],
    }


def validate_config(config: dict[str, Any]) -> None:
    conditions = config.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("config must contain at least one condition")
    names = [str(condition.get("name", "")) for condition in conditions]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("condition names must be non-empty and unique")
    if names[0] != "balanced_current":
        raise ValueError("the first condition must be balanced_current")
    if int(config.get("warmup_rows", 0)) < 1:
        raise ValueError("warmup_rows must be positive")
    if int(config.get("repeats", 0)) < 2:
        raise ValueError("repeats must be at least two")
    for condition in conditions:
        if condition.get("performance_mode") not in {"balanced", "throughput"}:
            raise ValueError(f"invalid performance mode in {condition!r}")
        batch_tokens = condition.get("max_num_batched_tokens")
        if batch_tokens is not None and int(batch_tokens) < 1:
            raise ValueError("max_num_batched_tokens must be positive or null")


def score_parity(
    baseline: list[float],
    candidate: list[float],
    *,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("score arrays must be non-empty and have equal length")
    deltas = [
        abs(left - right)
        for left, right in zip(baseline, candidate, strict=True)
    ]
    flips = sum(
        (left >= threshold) != (right >= threshold)
        for left, right in zip(baseline, candidate, strict=True)
    )
    return {
        "max_abs_score_delta": max(deltas),
        "mean_abs_score_delta": statistics.fmean(deltas),
        "threshold_flips": flips,
    }


def summarize_results(
    results: list[dict[str, Any]],
    parity_limits: dict[str, Any],
) -> dict[str, Any]:
    if not results or results[0]["condition"] != "balanced_current":
        raise ValueError("balanced_current must be the first successful result")
    baseline = results[0]
    baseline_rate = float(baseline["median_rows_per_second"])
    baseline_scores = [float(value) for value in baseline["scores"]]
    rows = []
    for result in results:
        parity = score_parity(baseline_scores, result["scores"])
        parity_passed = (
            parity["max_abs_score_delta"]
            <= float(parity_limits["max_abs_score_delta"])
            and parity["threshold_flips"]
            <= int(parity_limits["max_threshold_flips"])
        )
        rate = float(result["median_rows_per_second"])
        rows.append(
            {
                "condition": result["condition"],
                "median_seconds": result["median_seconds"],
                "median_rows_per_second": rate,
                "median_prompt_tokens_per_second": result[
                    "median_prompt_tokens_per_second"
                ],
                "speedup_vs_current": rate / baseline_rate,
                **parity,
                "parity_passed": parity_passed,
            }
        )
    eligible = [row for row in rows if row["parity_passed"]]
    best = max(eligible, key=lambda row: row["median_rows_per_second"])
    return {"baseline": baseline["condition"], "best": best["condition"], "rows": rows}
