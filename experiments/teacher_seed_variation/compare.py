"""Seed-to-seed, serving-drift and FP4/FA4 comparisons on ID and OOD."""

from __future__ import annotations

import statistics

from experiments.teacher_seed_variation.prepare import OOD, ROOT
from gleipnir.campaigns.monitoring.evaluation import attach_inputs, id_metrics
from gleipnir.data.monitoring import file_hash, read_rows, write_json

INPUTS = {"id": "data/id_cot_only_evaluation/prompts.jsonl", "ood": OOD["ood"][0]}
PREDICTIONS = {
    "sdpa_seed0_cached": {
        "id": "results/teacher-disagreement-delta000-sdpa04/evaluation/id.jsonl",
        "ood": "results/teacher-disagreement-strict-ood01/evaluation/ood.jsonl",
    },
    "sdpa_seed0_rescored": {
        s: f"results/teacher-seed-sdpa-s0-rescore01/evaluation/{s}.jsonl"
        for s in ("id", "ood")
    },
    "sdpa_seed1": {
        "id": "results/teacher-seed-sdpa-s1-01/evaluation/id.jsonl",
        "ood": "results/teacher-seed-sdpa-s1-ood01/evaluation/ood.jsonl",
    },
    "sdpa_seed2": {
        "id": "results/teacher-seed-sdpa-s2-01/evaluation/id.jsonl",
        "ood": "results/teacher-seed-sdpa-s2-ood01/evaluation/ood.jsonl",
    },
    "fp4_fa4_seed0_pass1": {
        "id": "results/filtered-fp4-fa4-bf16-serving01/evaluation/id.jsonl",
        "ood": "results/filtered-fp4-ood01/evaluation/repeat1/ood.jsonl",
    },
    "fp4_fa4_seed0_pass2": {
        "ood": "results/filtered-fp4-ood01/evaluation/repeat2/ood.jsonl",
    },
}
PAIRS = {
    "serving_drift_seed0": ("sdpa_seed0_rescored", "sdpa_seed0_cached"),
    "seed1_minus_seed0": ("sdpa_seed1", "sdpa_seed0_rescored"),
    "seed2_minus_seed0": ("sdpa_seed2", "sdpa_seed0_rescored"),
    "seed2_minus_seed1": ("sdpa_seed2", "sdpa_seed1"),
    "fp4_fa4_serving_repeat": ("fp4_fa4_seed0_pass2", "fp4_fa4_seed0_pass1"),
    "fp4_fa4_minus_sdpa_seed0": ("fp4_fa4_seed0_pass1", "sdpa_seed0_rescored"),
}


def load(path: str, split: str) -> list[dict]:
    return attach_inputs(read_rows(ROOT / path), read_rows(ROOT / INPUTS[split]))


def summary(rows: list[dict]) -> dict:
    metrics = id_metrics(rows)["metrics"]
    return {
        "macro": metrics["macro"]["macro"],
        "pooled": metrics["pooled"],
        "by_source": {
            g["group"]: g["pauroc_at_20"] for g in metrics["macro"]["groups"]
        },
    }


def score_difference(a: list[dict], b: list[dict]) -> dict:
    if [r["id"] for r in a] != [r["id"] for r in b]:
        raise ValueError("prediction membership differs")
    diffs = [abs(x["score"] - y["score"]) for x, y in zip(a, b, strict=True)]
    flips = sum(
        (x["score"] >= 0.5) != (y["score"] >= 0.5) for x, y in zip(a, b, strict=True)
    )
    return {
        "bitwise_equal": all(d == 0 for d in diffs),
        "mae": statistics.fmean(diffs),
        "max": max(diffs),
        "threshold_flips": flips,
    }


def compare() -> dict:
    rows = {
        model: {
            split: load(path, split)
            for split, path in paths.items()
            if (ROOT / path).is_file()
        }
        for model, paths in PREDICTIONS.items()
    }
    models = {
        model: {split: summary(r) for split, r in splits.items()}
        for model, splits in rows.items()
    }
    pairs = {}
    for name, (a, b) in PAIRS.items():
        pairs[name] = {}
        for split in sorted(set(rows[a]) & set(rows[b])):
            ma, mb = models[a][split], models[b][split]
            pairs[name][split] = {
                "macro_pauroc_at_20_delta": ma["macro"]["pauroc_at_20"]
                - mb["macro"]["pauroc_at_20"],
                "macro_auroc_delta": ma["macro"]["auroc"] - mb["macro"]["auroc"],
                "by_source_pauroc_delta": {
                    s: ma["by_source"][s] - mb["by_source"][s]
                    for s in ma["by_source"]
                    if ma["by_source"][s] is not None
                    and mb["by_source"].get(s) is not None
                },
                "scores": score_difference(rows[a][split], rows[b][split]),
            }
    seeds = ["sdpa_seed0_rescored", "sdpa_seed1", "sdpa_seed2"]
    spread = {}
    for split in ("id", "ood"):
        present = [m for m in seeds if split in models[m]]
        if len(present) < 2:
            continue
        sources = models[present[0]][split]["by_source"]
        spread[split] = {
            "seeds": present,
            **{
                key: {
                    "mean": statistics.fmean(values),
                    "sd": statistics.stdev(values),
                    "range": max(values) - min(values),
                }
                for key, values in [
                    (
                        "macro_pauroc_at_20",
                        [models[m][split]["macro"]["pauroc_at_20"] for m in present],
                    ),
                    *[
                        (s, [models[m][split]["by_source"][s] for m in present])
                        for s in sources
                        if sources[s] is not None
                    ],
                ]
            },
        }
    report = {
        "seed_spread": spread,
        "models": models,
        "pairs": pairs,
        "predictions_sha256": {
            p: file_hash(ROOT / p)
            for paths in PREDICTIONS.values()
            for p in paths.values()
            if (ROOT / p).is_file()
        },
    }
    out = ROOT / "results/teacher_seed_variation/comparison.json"
    write_json(out, report)
    for name, splits in pairs.items():
        for split, value in sorted(splits.items()):
            scores = value["scores"]
            print(
                f"{name:28} {split:3} "
                f"dpAUROC {value['macro_pauroc_at_20_delta']:+.5f} "
                f"dAUROC {value['macro_auroc_delta']:+.5f} "
                f"MAE {scores['mae']:.5f} flips {scores['threshold_flips']} "
                f"bitwise {scores['bitwise_equal']}"
            )
    return report


if __name__ == "__main__":
    compare()
