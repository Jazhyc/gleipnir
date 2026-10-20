"""Offline Gleipnir 4B → 27B cascade strategies, tuned on ID only.

Every row is scored by Gleipnir 4B v0.3 (optimized serving); a routing rule
escalates some rows to Gleipnir 27B (optimized serving, full teacher prompt),
and a merge rule turns the two scores into the cascade score. Cost per row is
the 4B prompt at its self-hosted break-even rate plus, when escalated, the
27B teacher prompt at its rate, both measured on the benchmark workload (one
B200 at $7.99/h, 100% utilization). Only the 3,012 ID rows are used here; the
OOD suite is reserved for the strategies chosen from these results.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from experiments.teacher_seed_variation.compare import load, summary
from gleipnir.data.monitoring import read_rows, write_json

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/cascade"
HOURLY = 7.99
# Benchmark-reference optimized-serving throughput (prompt tokens/s).
THROUGHPUT = {"4b": 225998.0, "27b": 53929.6}
INPUTS = {
    "4b": "results/qad-v02-optimized01/evaluation/id.jsonl",
    "27b": "results/optimized_gap/opt27b_teacher_id_eu01/id.jsonl",
    "4b_tokens": "results/b200_optimized_id/id02/workload.json",
    "27b_tokens": "data/optimized_gap/teacher_id_workload.jsonl",
}
FRACTIONS = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def load_inputs():
    rows = load(INPUTS["4b"], "id")
    big = {r["id"]: r["score"] for r in read_rows(ROOT / INPUTS["27b"])}
    small_tokens = {
        r["id"]: r["prompt_tokens"]
        for r in json.loads((ROOT / INPUTS["4b_tokens"]).read_text())
    }
    big_tokens = {
        r["id"]: r["prompt_tokens"] for r in read_rows(ROOT / INPUTS["27b_tokens"])
    }
    ids = [r["id"] for r in rows]
    if set(ids) != set(big) or set(ids) != set(small_tokens) != set(big_tokens):
        raise ValueError("ID coverage differs between inputs")
    p4 = np.array([r["score"] for r in rows])
    p27 = np.array([big[i] for i in ids])
    rate = {k: HOURLY / 3600 / v for k, v in THROUGHPUT.items()}
    cost4 = np.array([small_tokens[i] for i in ids]) * rate["4b"]
    cost27 = np.array([big_tokens[i] for i in ids]) * rate["27b"]
    return rows, p4, p27, cost4, cost27


# Routing rules: an order over rows; the first ``k`` are escalated.
def route_uncertain(p4):
    """Smallest 4B logit margin (closest to the 0.5 boundary) first."""
    return np.argsort(np.abs(logit(p4)), kind="stable")


def route_top(p4):
    """Highest 4B score first: the low-FPR region pAUROC@20 weighs."""
    return np.argsort(-p4, kind="stable")


ROUTES = {"uncertain": route_uncertain, "top": route_top}


def merge_replace(p4, p27, escalated):
    out = p4.copy()
    out[escalated] = p27[escalated]
    return out


def merge_logit_average(p4, p27, escalated):
    out = p4.copy()
    mixed = (logit(p4[escalated]) + logit(p27[escalated])) / 2
    out[escalated] = 1 / (1 + np.exp(-mixed))
    return out


def merge_reorder(p4, p27, escalated):
    """Keep the escalated set's 4B score values; reorder them by the 27B."""
    out = p4.copy()
    index = np.flatnonzero(escalated)
    values = np.sort(p4[index])
    out[index[np.argsort(p27[index], kind="stable")]] = values
    return out


MERGES = {
    "replace": merge_replace,
    "logit_average": merge_logit_average,
    "reorder": merge_reorder,
}


def evaluate(rows, scores, subset=None) -> dict:
    chosen = range(len(rows)) if subset is None else subset
    scored = [{**rows[i], "score": float(scores[i])} for i in chosen]
    metrics = summary(scored)
    return {
        "pauroc_at_20": metrics["macro"]["pauroc_at_20"],
        "auroc": metrics["macro"]["auroc"],
        "brier": metrics["macro"]["brier"],
        "by_source": metrics["by_source"],
    }


def escalation_mask(order, fraction, n, subset=None):
    mask = np.zeros(n, dtype=bool)
    pool = order if subset is None else [i for i in order if i in subset]
    mask[pool[: int(round(fraction * len(pool)))]] = True
    return mask


def band_orders(p4):
    """Two-sided bands: escalate 4B scores within [lo, hi] quantiles."""
    ranks = np.argsort(np.argsort(p4, kind="stable"), kind="stable") / (len(p4) - 1)
    bands = []
    for lo in (0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8):
        for hi in (0.6, 0.7, 0.8, 0.9, 0.95, 1.0):
            if hi > lo:
                bands.append((lo, hi, (ranks >= lo) & (ranks <= hi)))
    return bands


def sweep(rows, p4, p27, cost4, cost27):
    n = len(rows)
    results = []
    base = {
        "4b_only": evaluate(rows, p4),
        "27b_only": evaluate(rows, p27),
    }
    for name, value in base.items():
        cost = cost4 if name == "4b_only" else cost27
        results.append(
            {
                "strategy": name,
                "fraction": 0.0 if name == "4b_only" else 1.0,
                "cost_per_1k": 1000 * float(cost.mean()),
                **value,
            }
        )
    for route, rule in ROUTES.items():
        order = list(rule(p4))
        for fraction in FRACTIONS:
            mask = escalation_mask(order, fraction, n)
            cost = 1000 * float((cost4 + mask * cost27).mean())
            for merge, combine in MERGES.items():
                results.append(
                    {
                        "strategy": f"{route}/{merge}",
                        "fraction": float(mask.mean()),
                        "cost_per_1k": cost,
                        **evaluate(rows, combine(p4, p27, mask)),
                    }
                )
    for lo, hi, mask in band_orders(p4):
        cost = 1000 * float((cost4 + mask * cost27).mean())
        for merge, combine in MERGES.items():
            results.append(
                {
                    "strategy": f"band[{lo:.1f},{hi:.2f}]/{merge}",
                    "fraction": float(mask.mean()),
                    "cost_per_1k": cost,
                    **evaluate(rows, combine(p4, p27, mask)),
                }
            )
    return results


def cross_validate(rows, p4, p27, cost4, cost27, margin, repeats=5):
    """Pick the cheapest within-margin fraction on one half, score the other."""
    n = len(rows)
    strata = {}
    for i, r in enumerate(rows):
        strata.setdefault((r["source_dataset"], int(r["ground_truth"])), []).append(i)
    outcomes = {}
    for repeat in range(repeats):
        rng = random.Random(repeat)
        a, b = set(), set()
        for members in strata.values():
            members = members[:]
            rng.shuffle(members)
            a.update(members[: len(members) // 2])
            b.update(members[len(members) // 2 :])
        for tune, test in ((a, b), (b, a)):
            target = evaluate(rows, p27, sorted(tune))["pauroc_at_20"] - margin
            test_big = evaluate(rows, p27, sorted(test))["pauroc_at_20"]
            for route, rule in ROUTES.items():
                order = list(rule(p4))
                for merge, combine in MERGES.items():
                    chosen = 1.0
                    for fraction in FRACTIONS:
                        mask = escalation_mask(order, fraction, n, tune)
                        tuned = evaluate(rows, combine(p4, p27, mask), sorted(tune))
                        if tuned["pauroc_at_20"] >= target:
                            chosen = fraction
                            break
                    mask = escalation_mask(order, chosen, n, test)
                    held = evaluate(rows, combine(p4, p27, mask), sorted(test))
                    cost = 1000 * float((cost4 + mask * cost27)[sorted(test)].mean())
                    outcomes.setdefault(f"{route}/{merge}", []).append(
                        {
                            "fraction": chosen,
                            "held_out_pauroc": held["pauroc_at_20"],
                            "held_out_gap_to_27b": test_big - held["pauroc_at_20"],
                            "held_out_cost_per_1k": cost,
                        }
                    )
    return {
        k: {
            "mean_fraction": float(np.mean([o["fraction"] for o in v])),
            "mean_held_out_pauroc": float(np.mean([o["held_out_pauroc"] for o in v])),
            "mean_gap_to_27b": float(np.mean([o["held_out_gap_to_27b"] for o in v])),
            "max_gap_to_27b": float(np.max([o["held_out_gap_to_27b"] for o in v])),
            "mean_cost_per_1k": float(np.mean([o["held_out_cost_per_1k"] for o in v])),
        }
        for k, v in outcomes.items()
    }


def main() -> None:
    rows, p4, p27, cost4, cost27 = load_inputs()
    results = sweep(rows, p4, p27, cost4, cost27)
    big = next(r for r in results if r["strategy"] == "27b_only")["pauroc_at_20"]
    selections = {}
    for margin in (0.005, 0.01, 0.02):
        best = {}
        for r in results:
            if r["pauroc_at_20"] >= big - margin and (
                r["strategy"] not in best
                or r["cost_per_1k"] < best[r["strategy"]]["cost_per_1k"]
            ):
                best[r["strategy"]] = r
        selections[str(margin)] = sorted(best.values(), key=lambda r: r["cost_per_1k"])
    report = {
        "inputs": INPUTS,
        "throughput": THROUGHPUT,
        "hourly": HOURLY,
        "rows": len(rows),
        "results": results,
        "cheapest_within_margin": selections,
        "cross_validation": {
            str(m): cross_validate(rows, p4, p27, cost4, cost27, m)
            for m in (0.005, 0.01)
        },
    }
    write_json(OUT / "id_sweep.json", report)
    print("cascade_id_sweep", len(results), "conditions", flush=True)


if __name__ == "__main__":
    main()
