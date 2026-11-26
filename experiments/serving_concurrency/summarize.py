"""Summarize the timed passes of every condition."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/serving_concurrency"
PATTERN = re.compile(r"(full|quick)_c(\d+)_repeat(\d+)\.json")


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile of ``values`` (``q`` in [0, 1])."""
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def pass_summary(rows: list[dict], seconds: float) -> dict:
    latencies = [row["latency_seconds"] for row in rows]
    return {
        "prompt_tokens_per_second": sum(r["prompt_tokens"] for r in rows) / seconds,
        "requests_per_second": len(rows) / seconds,
        "p50_ms": 1000 * percentile(latencies, 0.5),
        "p95_ms": 1000 * percentile(latencies, 0.95),
    }


def condition_summary(directory: Path) -> dict:
    passes: dict[tuple[str, int], list[dict]] = {}
    for path in sorted(directory.glob("*_repeat*.json")):
        match = PATTERN.fullmatch(path.name)
        if not match:
            continue
        data = json.loads(path.read_text())
        key = (match.group(1), int(match.group(2)))
        passes.setdefault(key, []).append(pass_summary(data["rows"], data["seconds"]))
    summary = {}
    for (workload, concurrency), items in sorted(passes.items()):
        tokens = [item["prompt_tokens_per_second"] for item in items]
        summary[f"{workload}_c{concurrency}"] = {
            "passes": len(items),
            "prompt_tokens_per_second": statistics.median(tokens),
            "prompt_tokens_per_second_range": [min(tokens), max(tokens)],
            "p50_ms": statistics.median(item["p50_ms"] for item in items),
            "p95_ms": statistics.median(item["p95_ms"] for item in items),
        }
    lengths = directory / "lengths.json"
    if lengths.exists():
        summary["lengths"] = {
            str(item["prompt_tokens"]): {
                "median_ms": 1000 * statistics.median(item["latency_seconds"]),
                "min_ms": 1000 * min(item["latency_seconds"]),
                "max_ms": 1000 * max(item["latency_seconds"]),
            }
            for item in json.loads(lengths.read_text())["measurements"]
            if "latency_seconds" in item
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default="", help="output subdirectory")
    out = OUT / parser.parse_args().session
    report = {
        directory.name: condition_summary(directory)
        for directory in sorted(out.iterdir())
        if (directory / "condition.json").exists()
    }
    (out / "summary.json").write_text(json.dumps(report, indent=1) + "\n")
    for name, summary in report.items():
        for key, item in summary.items():
            if key == "lengths":
                for length, value in item.items():
                    print(
                        f"{name:14s} length {length:>6s} {value['median_ms']:8.1f} ms"
                    )
                continue
            print(
                f"{name:14s} {key:10s} {item['prompt_tokens_per_second']:9.0f} tok/s"
                f"  p50 {item['p50_ms']:8.1f} ms  p95 {item['p95_ms']:8.1f} ms"
            )


if __name__ == "__main__":
    main()
