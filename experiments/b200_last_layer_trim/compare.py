"""Compare a trim run with the archived 1M baseline: exactness, speed, clocks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

# Baseline fit (docs/findings/b200_context_million.md): ms per 32K x 32K block.
ATTENTION_BLOCK_MS = 71.8
CHUNK = 32768


def load(directory: Path, phase: str, length: int) -> list[dict]:
    return [
        json.loads((directory / phase / f"{length}_repeat{r}.json").read_text())
        for r in range(5)
    ]


def predicted_saving_ms(length: int) -> float | None:
    if length < CHUNK or length % CHUNK:
        return None
    n = length // CHUNK
    return ATTENTION_BLOCK_MS / 8 * n * (n + 1) / 2


def compare(candidate: Path, baseline: Path) -> dict:
    report = json.loads((candidate / "summary.json").read_text())
    control = json.loads((baseline / "summary.json").read_text())
    if report["status"] != "complete" or control["status"] != "complete":
        raise ValueError("both runs must be complete")
    if (candidate / "workload.json").read_bytes() != (
        baseline / "workload.json"
    ).read_bytes():
        raise ValueError("workloads differ")
    rows, exact = [], True
    for phase in report["phases"]:
        name = phase["phase"]
        reference = next(p for p in control["phases"] if p["phase"] == name)
        canary = [
            json.loads((d / name / "canary_predictions.json").read_text())
            for d in (candidate, baseline)
        ]
        canary_equal = [r["score"] for r in canary[0]] == [
            r["score"] for r in canary[1]
        ]
        exact &= canary_equal
        for m in phase["measurements"]:
            length = m["prompt_tokens"]
            base = next(
                r for r in reference["measurements"] if r["prompt_tokens"] == length
            )
            scores = [t["values"][0] for t in load(candidate, name, length)]
            control_scores = [t["values"][0] for t in load(baseline, name, length)]
            difference = max(
                abs(a["score"] - b["score"]) for a in scores for b in control_scores
            )
            margins_equal = {a["margin"] for a in scores} == {
                b["margin"] for b in control_scores
            }
            equal = difference == 0.0 and margins_equal
            exact &= equal
            saving = 1000 * (
                base["median_latency_seconds"] - m["median_latency_seconds"]
            )
            predicted = predicted_saving_ms(length)
            rows.append(
                {
                    "phase": name,
                    "prompt_tokens": length,
                    "baseline_latency_ms": 1000 * base["median_latency_seconds"],
                    "trim_latency_ms": 1000 * m["median_latency_seconds"],
                    "latency_ratio": m["median_latency_seconds"]
                    / base["median_latency_seconds"],
                    "saving_ms": saving,
                    "predicted_saving_ms": predicted,
                    "baseline_tokens_per_second": base["prompt_tokens_per_second"],
                    "trim_tokens_per_second": m["prompt_tokens_per_second"],
                    "baseline_sm_clock_mhz": base["gpu_samples"].get(
                        "sm_clock_median_mhz"
                    ),
                    "trim_sm_clock_mhz": m["gpu_samples"].get("sm_clock_median_mhz"),
                    "baseline_power_max_w": base["gpu_samples"].get("power_max_w"),
                    "trim_power_max_w": m["gpu_samples"].get("power_max_w"),
                    "scores_bitwise_equal": equal,
                    "max_score_difference": difference,
                }
            )
        rows.append({"phase": name, "canary_scores_bitwise_equal": canary_equal})
    result = {
        "candidate": str(candidate),
        "baseline": str(baseline),
        "all_scores_bitwise_equal": exact,
        "rows": rows,
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (candidate / "summary.json", baseline / "summary.json")
        },
        "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (candidate / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument(
        "--baseline", type=Path, default=Path("results/b200_context_million/million01")
    )
    args = parser.parse_args()
    result = compare(args.candidate, args.baseline)
    print("all_scores_bitwise_equal", result["all_scores_bitwise_equal"])
    for row in result["rows"]:
        if "prompt_tokens" not in row:
            print(row)
            continue
        predicted = row["predicted_saving_ms"]
        print(
            f"{row['phase']:6} {row['prompt_tokens']:>8} "
            f"{row['baseline_latency_ms']:10.2f} -> {row['trim_latency_ms']:10.2f} ms "
            f"x{row['latency_ratio']:.4f} saved {row['saving_ms']:8.1f} "
            f"(pred {predicted if predicted is None else round(predicted, 1)}) "
            f"clk {row['baseline_sm_clock_mhz']}->{row['trim_sm_clock_mhz']} "
            f"exact={row['scores_bitwise_equal']} d={row['max_score_difference']:.2e}"
        )


if __name__ == "__main__":
    main()
