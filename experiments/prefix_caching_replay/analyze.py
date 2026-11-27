"""Summarize a per-action replay: time, cost and score drift per model.

``python -m experiments.prefix_caching_replay.analyze --name replay01``
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()
)
OUT = ROOT / "results/prefix_caching_replay"
RATES = {"on_demand": 7.99, "committed": 3.44}


def load(path: Path) -> dict[str, dict]:
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines())}


def per_thousand(seconds: float, trajectories: int, rate: float) -> float:
    return seconds / trajectories * 1000 * rate / 3600


def summarize(name: str, model: str) -> dict:
    cached = load(OUT / name / f"{model}_cached.jsonl")
    uncached = load(OUT / name / f"{model}_uncached.jsonl")
    ids = sorted(set(cached) & set(uncached))
    if len(ids) != 96:
        raise ValueError(f"{model}: {len(ids)} trajectories in both modes")
    seconds = {"isolated": 0.0, "uncached": 0.0, "cached": 0.0}
    tokens = {"isolated": 0, "uncached": 0, "cached_hit": 0, "cached_queried": 0}
    drift, ratios = [], []
    for i in ids:
        c, u = cached[i]["checks"], uncached[i]["checks"]
        if len(c) != len(u):
            raise ValueError(f"{i}: check counts differ")
        iso = uncached[i]["isolated"]["latency_seconds"]
        t_c = sum(v["latency_seconds"] for v in c)
        t_u = sum(v["latency_seconds"] for v in u)
        seconds["isolated"] += iso
        seconds["cached"] += t_c
        seconds["uncached"] += t_u
        ratios.append(t_c / iso)
        tokens["isolated"] += uncached[i]["isolated"]["prompt_tokens"]
        tokens["uncached"] += sum(v["prompt_tokens"] for v in u)
        tokens["cached_hit"] += cached[i]["cache_hit_tokens"]
        tokens["cached_queried"] += cached[i]["cache_queried_tokens"]
        drift += [abs(a["score"] - b["score"]) for a, b in zip(c, u, strict=True)]
    return {
        "trajectories": len(ids),
        "checks": len(drift),
        "seconds": seconds,
        "cached_over_isolated": seconds["cached"] / seconds["isolated"],
        "uncached_over_isolated": seconds["uncached"] / seconds["isolated"],
        "uncached_over_cached": seconds["uncached"] / seconds["cached"],
        "cached_over_isolated_median_trajectory": statistics.median(ratios),
        "tokens": tokens,
        "cache_hit_fraction": tokens["cached_hit"] / max(tokens["cached_queried"], 1),
        "cost_per_1k_trajectories": {
            policy: {
                rate: per_thousand(seconds[policy], len(ids), usd)
                for rate, usd in RATES.items()
            }
            for policy in seconds
        },
        "score_drift": {
            "mean": statistics.fmean(drift),
            "median": statistics.median(drift),
            "max": max(drift),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="replay01")
    args = parser.parse_args()
    result = {m: summarize(args.name, m) for m in ("4b", "27b")}
    (OUT / args.name / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
