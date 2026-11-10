"""Tabulate the 4B serving ladder and project cost per 1,000 evaluations."""

from __future__ import annotations

import json
import math
import statistics

from experiments.gleipnir4b_serving_ladder.run import OUT, RUNGS

OOD_TOKENS_PER_EVALUATION = 60_539_619 / 6_395
HOURLY_RATES = {"on_demand": 7.99, "committed": 3.44}


def cost_per_thousand(tokens_per_second: float, hourly_rate: float) -> float:
    """USD per 1,000 evaluations on one fully used GPU."""
    if tokens_per_second <= 0:
        raise ValueError("throughput must be positive")
    seconds = 1000 * OOD_TOKENS_PER_EVALUATION / tokens_per_second
    return seconds * hourly_rate / 3600


def throughput(rows: list[dict], seconds: float) -> float:
    return sum(r["prompt_tokens"] for r in rows) / seconds


def score_mae(rows: list[dict], reference: list[dict]) -> float:
    by_id = {r["id"]: r["score"] for r in reference}
    if set(by_id) != {r["id"] for r in rows}:
        raise ValueError("row identities differ")
    values = [abs(r["score"] - by_id[r["id"]]) for r in rows]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("non-finite score")
    return sum(values) / len(values)


def summarize(repeats: dict[str, list[tuple[list[dict], float]]]) -> dict:
    """``repeats`` maps each rung to its c128 (rows, seconds) passes."""
    medians = {
        rung: statistics.median(throughput(rows, s) for rows, s in passes)
        for rung, passes in repeats.items()
    }
    first = next(iter(medians))
    last = next(reversed(medians))
    table = {}
    previous = None
    for rung, median in medians.items():
        table[rung] = {
            "prompt_tokens_per_second": [
                throughput(rows, s) for rows, s in repeats[rung]
            ],
            "median_prompt_tokens_per_second": median,
            "speedup_over_first": median / medians[first],
            "speedup_over_previous": None if previous is None else median / previous,
            "usd_per_1000": {
                name: cost_per_thousand(median, rate)
                for name, rate in HOURLY_RATES.items()
            },
            "score_mae_against_last": score_mae(
                repeats[rung][0][0], repeats[last][0][0]
            ),
        }
        previous = median
    return table


def load(rung: str) -> list[tuple[list[dict], float]]:
    passes = []
    for path in sorted((OUT / rung).glob("c128_repeat*.json")):
        payload = json.loads(path.read_text())
        passes.append((payload["rows"], payload["seconds"]))
    if not passes:
        raise FileNotFoundError(f"no c128 passes for {rung}")
    return passes


def main() -> None:
    table = summarize({rung: load(rung) for rung in RUNGS if (OUT / rung).exists()})
    (OUT / "comparison.json").write_text(json.dumps(table, indent=2) + "\n")
    for rung, row in table.items():
        print(
            f"{rung:10s} {row['median_prompt_tokens_per_second']:9.0f} tok/s  "
            f"{row['speedup_over_first']:.2f}x  "
            f"${row['usd_per_1000']['on_demand']:.4f} / "
            f"${row['usd_per_1000']['committed']:.4f}  "
            f"MAE {row['score_mae_against_last']:.4f}"
        )


if __name__ == "__main__":
    main()
