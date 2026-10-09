"""Describe fixed-axis training associations without claiming example influence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from gleipnir.data.monitoring import read_rows, write_json, write_rows


def paired_values(base: dict, trained: dict) -> dict:
    """Compare identical prompts in the unchanged residual coordinate system."""
    if (base["id"], base["prompt_sha256"], base["prompt_tokens"]) != (
        trained["id"],
        trained["prompt_sha256"],
        trained["prompt_tokens"],
    ):
        raise ValueError("model comparison identity drift")
    result = {
        k: v for k, v in trained.items() if k not in ["logits", "latency_seconds"]
    }
    for layer in [20, 31]:
        for name, r in [("base", base), ("trained", trained)]:
            result[f"{name}_z{layer}"] = r[f"z{layer}"]
            result[f"{name}_cos{layer}"] = r[f"z{layer}"] / r[f"norm{layer}"]
            result[f"{name}_norm{layer}"] = r[f"norm{layer}"]
        result[f"delta_z{layer}"] = trained[f"z{layer}"] - base[f"z{layer}"]
        result[f"delta_cos{layer}"] = (
            result[f"trained_cos{layer}"] - result[f"base_cos{layer}"]
        )
    result["base_score"] = base["score"]
    result["trained_score"] = trained["score"]
    result["score_delta"] = trained["score"] - base["score"]
    return result


def distribution(values: list[float]) -> dict:
    a = np.array(values, dtype=np.float64)
    if not len(a) or not np.isfinite(a).all():
        raise ValueError("invalid summary distribution")
    return {
        "mean": float(a.mean()),
        "mean_absolute": float(abs(a).mean()),
        "median": float(np.median(a)),
        "std": float(a.std()),
        "p10_p90": np.quantile(a, [0.1, 0.9]).tolist(),
        "fraction_positive": float((a > 0).mean()),
    }


def correlation(x: list[float], y: list[float]) -> float | None:
    if len(set(x)) < 2 or len(set(y)) < 2:
        return None
    return float(spearmanr(x, y).statistic)


def describe(rows: list[dict]) -> dict:
    fields = [
        f"{model}_{metric}{layer}"
        for layer in [20, 31]
        for model in ["base", "trained"]
        for metric in ["z", "cos", "norm"]
    ]
    fields += ["delta_z20", "delta_cos20", "delta_z31", "delta_cos31", "score_delta"]
    return {
        "rows": len(rows),
        **{f: distribution([r[f] for r in rows]) for f in fields},
        "correlations": {
            field: {
                "length": correlation(
                    [r[field] for r in rows], [r["prompt_tokens"] for r in rows]
                ),
                "teacher_target": correlation(
                    [r[field] for r in rows], [r["soft_target"] for r in rows]
                ),
                "trained_score": correlation(
                    [r[field] for r in rows], [r["trained_score"] for r in rows]
                ),
            }
            for field in ["base_z20", "trained_z20", "delta_z20", "delta_cos20"]
        },
        "hard_label_auroc": {
            field: float(
                roc_auc_score([r["label"] for r in rows], [r[field] for r in rows])
            )
            if len({r["label"] for r in rows}) == 2
            else None
            for field in ["base_z20", "trained_z20", "delta_z20"]
        },
    }


def analyze(out: Path, root: Path, c: dict) -> None:
    b = read_rows(out / "base.jsonl")
    t = read_rows(out / "trained.jsonl")
    if len(b) != len(t):
        raise ValueError("missing model rows")
    values = [paired_values(x, y) for x, y in zip(b, t, strict=True)]
    actual = [r for r in values if r["view"] == "actual"]
    lookup = {(r["dataset"], r["index"], r["view"]): r for r in values}
    pairs = []
    for r in actual:
        if not r["training_augmented"]:
            continue
        parent = lookup[r["dataset"], r["index"], "clean_parent"]
        item = dict(r)
        for layer in [20, 31]:
            for metric in ["z", "cos"]:
                for model in ["base", "trained"]:
                    f = f"{model}_{metric}{layer}"
                    item[f + "_injection_shift"] = r[f] - parent[f]
                item[f"did_{metric}{layer}"] = (
                    item[f"trained_{metric}{layer}_injection_shift"]
                    - item[f"base_{metric}{layer}_injection_shift"]
                )
        pairs.append(item)
    clean = [
        r for r in values if r["view"] == "clean_parent" or not r["training_augmented"]
    ]
    populations = {
        "actual_training": actual,
        "original_clean": clean,
        "augmented_training": [r for r in actual if r["training_augmented"]],
        "unaugmented_training": [r for r in actual if not r["training_augmented"]],
    }
    summary = {
        "populations": {k: describe(rs) for k, rs in populations.items()},
        "by_source": {
            s: describe([r for r in actual if r["source"] == s])
            for s in sorted({r["source"] for r in actual})
        },
        "by_label": {
            str(label): describe([r for r in actual if r["label"] == label])
            for label in [0, 1]
        },
        "paired_injection": {},
    }
    views = {"all": pairs}
    for field in [
        "source",
        "label",
        "template_id",
        "template_family",
        "role",
        "demanded_decision",
    ]:
        for value in sorted({str(r[field]) for r in pairs}):
            views[field + "/" + value] = [r for r in pairs if str(r[field]) == value]
    views["demand_agrees"] = [r for r in pairs if r["demanded_decision"] == r["label"]]
    views["demand_conflicts"] = [
        r for r in pairs if r["demanded_decision"] != r["label"]
    ]
    fields = [
        f"{model}_{metric}{layer}_injection_shift"
        for layer in [20, 31]
        for model in ["base", "trained"]
        for metric in ["z", "cos"]
    ]
    fields += [f"did_{metric}{layer}" for layer in [20, 31] for metric in ["z", "cos"]]
    summary["paired_injection"] = {
        k: {"rows": len(rs), **{f: distribution([r[f] for r in rs]) for f in fields}}
        for k, rs in views.items()
    }
    training = {
        (r["dataset"], str(r["index"])): r
        for r in read_rows(root / c["inputs"]["training"]["path"])
    }
    originals = {
        (r["dataset"], str(r["index"])): r
        for r in read_rows(root / c["inputs"]["clean"]["path"])
    }

    def excerpt(r):
        raw = (originals if r["view"] == "clean_parent" else training)[
            r["dataset"], r["index"]
        ]
        text = raw["student_prompt"]
        start = text.index("<agent_trajectory>") + len("<agent_trajectory>")
        text = text[start : text.rindex("</agent_trajectory>")]
        contexts = []
        if aug := raw.get("augmentation"):
            a = aug["offset"]
            b = a + aug["inserted_characters"]
            contexts.append(text[max(0, a - 250) : b + 250])
        for needle in [
            "ignore",
            "reviewer",
            "deceiv",
            "secret",
            "conceal",
            "Prediction:",
            "malicious",
        ]:
            i = text.lower().find(needle.lower())
            if i >= 0 and len(contexts) < 4:
                contexts.append(text[max(0, i - 180) : i + 300])
        return r | {
            "trajectory_start": text[:700],
            "trajectory_end": text[-700:],
            "context_excerpts": contexts,
        }

    rankings = {
        "trained_alignment_actual": (actual, "trained_z20"),
        "training_increase_actual": (actual, "delta_z20"),
        "training_increase_clean": (clean, "delta_z20"),
        "training_cosine_increase_actual": (actual, "delta_cos20"),
        "base_alignment_actual": (actual, "base_z20"),
        "augmentation_amplification": (pairs, "did_z20"),
    }
    for name, (rs, field) in rankings.items():
        for sign, reverse in [("high", True), ("low", False)]:
            ranked = sorted(
                rs, key=lambda r: ((-1 if reverse else 1) * r[field], r["id"])
            )[:20]
            write_rows(
                out / (name + "_" + sign + ".jsonl"), [excerpt(r) for r in ranked]
            )
    write_rows(out / "paired_models.jsonl", values)
    write_rows(out / "paired_injections.jsonl", pairs)
    timing = {}
    for model in ["base", "trained"]:
        batches = [
            json.loads(p.read_text()) for p in (out / "batches" / model).glob("*.json")
        ]
        seconds = sum(r["seconds"] for r in batches)
        tokens = sum(r["tokens"] for r in batches)
        latencies = [r["latency_seconds"] for r in (b if model == "base" else t)]
        timing[model] = {
            "seconds": seconds,
            "input_tokens": tokens,
            "input_tokens_per_second": tokens / seconds,
            "requests_per_second": len(values) / seconds,
            "latency_p50_p95_seconds": np.quantile(latencies, [0.5, 0.95]).tolist(),
            "latency_includes_semaphore_wait": True,
        }
    summary["timing"] = timing
    summary["qualification"] = (
        "Fixed 512-trajectory training-seen sample; "
        "alignment is not example influence. "
        "Both models share the trained-model axis; no refit or causal data ablation."
    )
    write_json(out / "summary.json", summary)
