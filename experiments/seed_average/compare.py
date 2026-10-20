"""Apply the frozen release rule: the average versus the three seeds, ID only.

OOD is reported descriptively for every model and never enters the decision.
"""

from __future__ import annotations

import statistics

from experiments.seed_average.prepare import AVERAGE, AVERAGE_OOD, OUT
from experiments.teacher_seed_variation.compare import load, score_difference, summary
from gleipnir.data.monitoring import write_json

SEEDS = {
    "seed0": {
        s: f"results/teacher-seed-sdpa-s0-rescore01/evaluation/{s}.jsonl"
        for s in ("id", "ood")
    },
    "seed1": {
        "id": "results/teacher-seed-sdpa-s1-01/evaluation/id.jsonl",
        "ood": "results/teacher-seed-sdpa-s1-ood01/evaluation/ood.jsonl",
    },
    "seed2": {
        "id": "results/teacher-seed-sdpa-s2-01/evaluation/id.jsonl",
        "ood": "results/teacher-seed-sdpa-s2-ood01/evaluation/ood.jsonl",
    },
}
AVERAGED = {
    "id": f"results/{AVERAGE}/evaluation/id.jsonl",
    "ood": f"results/{AVERAGE_OOD}/evaluation/ood.jsonl",
}


def decide(average_id: float, seed_ids: dict[str, float]) -> dict:
    mean = statistics.fmean(seed_ids.values())
    best = max(seed_ids, key=seed_ids.get)
    accept = average_id >= mean
    return {
        "rule": "average if ID macro pAUROC@20 >= three-seed mean, else ID-best seed",
        "seed_mean": mean,
        "average": average_id,
        "release_candidate": "average" if accept else best,
    }


def compare() -> dict:
    models = {**SEEDS, "average": AVERAGED}
    rows = {m: {s: load(p, s) for s, p in paths.items()} for m, paths in models.items()}
    metrics = {m: {s: summary(r) for s, r in v.items()} for m, v in rows.items()}
    macro = {
        s: {m: metrics[m][s]["macro"]["pauroc_at_20"] for m in models}
        for s in ("id", "ood")
    }
    seed_ids = {m: macro["id"][m] for m in SEEDS}
    report = {
        "decision": decide(macro["id"]["average"], seed_ids),
        "macro_pauroc_at_20": macro,
        "seed_mean": {
            s: statistics.fmean(macro[s][m] for m in SEEDS) for s in ("id", "ood")
        },
        "metrics": metrics,
        "average_vs_seeds": {
            m: {s: score_difference(rows["average"][s], rows[m][s]) for s in rows[m]}
            for m in SEEDS
        },
    }
    write_json(OUT / "comparison.json", report)
    print("seed_average_decision", report["decision"], flush=True)
    return report


if __name__ == "__main__":
    compare()
