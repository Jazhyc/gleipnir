"""Frozen workload selection and strict binary HTTP inference measurements."""

from __future__ import annotations

import math
import random
import statistics
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


def paired_score_summary(
    baseline: list[list[dict]], candidate: list[list[dict]]
) -> dict:
    """Compare repeat-median scores, requiring identical ordered prompt identities."""
    if not baseline or not candidate or not baseline[0]:
        raise ValueError("empty paired measurement")
    identities = [(r["id"], r["prompt_sha256"]) for r in baseline[0]]
    for rows in baseline + candidate:
        if [(r["id"], r["prompt_sha256"]) for r in rows] != identities:
            raise ValueError("paired prompt identity drift")
        if not all(math.isfinite(r[k]) for r in rows for k in ("score", "margin")):
            raise ValueError("nonfinite paired score")
    medians = [
        {
            key: [
                statistics.median(r[i][key] for r in runs)
                for i in range(len(identities))
            ]
            for key in ("score", "margin")
        }
        for runs in (baseline, candidate)
    ]
    differences = {}
    for key in ("score", "margin"):
        errors = [
            abs(a - b) for a, b in zip(medians[0][key], medians[1][key], strict=True)
        ]
        differences[key] = {
            "mean_absolute_difference": statistics.mean(errors),
            "max_absolute_difference": max(errors),
        }
    flips = [
        identity
        for identity, a, b in zip(
            identities, medians[0]["score"], medians[1]["score"], strict=True
        )
        if (a >= 0.5) != (b >= 0.5)
    ]
    return {
        **differences,
        "threshold_flips": len(flips),
        "threshold_flip_ids": [identity[0] for identity in flips],
        "baseline_threshold_unstable_ids": [
            identities[i][0]
            for i in range(len(identities))
            if len({rows[i]["score"] >= 0.5 for rows in baseline}) > 1
        ],
    }


def ranking_comparison(
    rows: list[dict], baseline: list[list[dict]], candidate: list[list[dict]]
) -> dict:
    """Report matched development ranking/calibration metrics and AUROC deltas."""
    import pandas as pd

    from gleipnir.binary_evaluation import metric_views

    paired_score_summary(baseline, candidate)
    if [(r["id"], r["prompt_sha256"]) for r in rows] != [
        (r["id"], r["prompt_sha256"]) for r in baseline[0]
    ]:
        raise ValueError("ranking label/prompt identity drift")

    def metrics(scores: list[float]) -> dict:
        return metric_views(
            pd.DataFrame(
                [
                    {"dataset": r["dataset"], "label": r["label"], "score": score}
                    for r, score in zip(rows, scores, strict=True)
                ]
            )
        )

    medians, repeats = [], []
    for runs in (baseline, candidate):
        medians.append(
            metrics(
                [
                    statistics.median(output[i]["score"] for output in runs)
                    for i in range(len(rows))
                ]
            )
        )
        repeats.append([metrics([r["score"] for r in output]) for output in runs])
    groups = [{g["group"]: g["auroc"] for g in m["macro"]["groups"]} for m in medians]

    def delta(a, b):
        return b - a if a is not None and b is not None else None

    return {
        "population": "frozen training-seen systems development workload",
        "baseline_repeat_median_scores": medians[0],
        "candidate_repeat_median_scores": medians[1],
        "repeat_metrics": {"baseline": repeats[0], "candidate": repeats[1]},
        "auroc_delta": {
            "pooled": delta(
                medians[0]["pooled"]["auroc"], medians[1]["pooled"]["auroc"]
            ),
            "macro": delta(
                medians[0]["macro"]["macro"]["auroc"],
                medians[1]["macro"]["macro"]["auroc"],
            ),
            "per_source": {
                name: delta(value, groups[1][name]) for name, value in groups[0].items()
            },
        },
        "macro_auroc_sources": [name for name, v in groups[0].items() if v is not None],
        "undefined_auroc_sources": [name for name, v in groups[0].items() if v is None],
    }
