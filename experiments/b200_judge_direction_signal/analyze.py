"""Fixed-direction paired diagnostics; no fitted classifier or independent-row tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from gleipnir.data.monitoring import read_rows, write_json
from gleipnir.evaluation.preferences import summarize_preferences


def shift(values: list[float]) -> dict:
    array = np.array(values)
    return {
        "rows": len(values),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "quantiles_05_25_75_95": np.quantile(array, [0.05, 0.25, 0.75, 0.95]).tolist(),
        "fraction_positive": float(np.mean(array > 0)),
        "fraction_zero": float(np.mean(array == 0)),
        "paired_win_rate": float(np.mean((array > 0) + 0.5 * (array == 0))),
    }


def detection(injected: list[dict], clean: dict[str, dict]) -> dict:
    negatives = [clean[r["clean_id"]]["z20"] for r in injected]
    positives = [r["z20"] for r in injected]
    return {
        "paired_shift": shift(
            [a - b for a, b in zip(positives, negatives, strict=True)]
        ),
        "auroc": float(
            roc_auc_score(
                [0] * len(negatives) + [1] * len(positives), negatives + positives
            )
        ),
        "unique_clean_parents": len({r["clean_id"] for r in injected}),
    }


def correlation(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(spearmanr(a, b).statistic)


def analyze(out: Path) -> None:
    plain = read_rows(out / "plain.jsonl")
    project = read_rows(out / "project.jsonl")
    controls = {
        kind: {r["attack_id"]: r for r in read_rows(out / (kind + ".jsonl"))}
        for kind in ["natural", "whitespace"]
    }
    clean = {r["id"]: r for r in plain if r["condition"] == "clean"}
    projected = {r["id"]: r for r in project}
    injected = [r for r in plain if r["condition"] != "clean"]
    views = {"all": injected}
    for field in ["condition", "source", "lineage_group"]:
        views.update(
            {
                field + "/" + str(value): [r for r in injected if r[field] == value]
                for value in sorted({r[field] for r in injected})
            }
        )
    result = {
        "primary_layer": 20,
        "direction_sign": "fixed APPS injected-minus-neutral",
        "diagnostics": {},
        "qualification": (
            "Matched-clean-weighted ROC; six queries and dependent variants. "
            "Benign copied labels are bookkeeping, not new preference truth. "
            "Correlations are descriptive, without row-independent p-values."
        ),
    }
    for name, rs in views.items():
        report = {"attack_vs_clean": detection(rs, clean), "controls": {}}
        for kind, lookup in controls.items():
            ctrl = [lookup[r["id"]] for r in rs]
            report["controls"][kind] = {
                "control_vs_clean": detection(ctrl, clean),
                "attack_minus_control": shift(
                    [r["z20"] - c["z20"] for r, c in zip(rs, ctrl, strict=True)]
                ),
                "attack_vs_control_auroc": float(
                    roc_auc_score(
                        [0] * len(ctrl) + [1] * len(rs),
                        [c["z20"] for c in ctrl] + [r["z20"] for r in rs],
                    )
                ),
            }
        dz = [r["z20"] - clean[r["clean_id"]]["z20"] for r in rs]
        dm = [
            r["correct_margin_ab"] - clean[r["clean_id"]]["correct_margin_ab"]
            for r in rs
        ]
        intervention = [
            projected[r["id"]]["correct_margin_ab"] - r["correct_margin_ab"] for r in rs
        ]
        report["decision_coupling"] = {
            "rho_injection_shift_vs_injection_correct_margin_change": correlation(
                dz, dm
            ),
            "rho_injection_shift_vs_projection_correct_margin_change": correlation(
                dz, intervention
            ),
            "projection_correct_margin_change": shift(intervention),
            "injection_correct_margin_change": shift(dm),
        }
        result["diagnostics"][name] = report
    for field in ["source", "lineage_group"]:
        group = [
            v for k, v in result["diagnostics"].items() if k.startswith(field + "/")
        ]
        result[field + "_macro"] = {
            "groups": len(group),
            "attack_vs_clean_auroc": float(
                np.mean([v["attack_vs_clean"]["auroc"] for v in group])
            ),
            "attack_shift_mean": float(
                np.mean([v["attack_vs_clean"]["paired_shift"]["mean"] for v in group])
            ),
            "attack_positive_fraction": float(
                np.mean(
                    [
                        v["attack_vs_clean"]["paired_shift"]["fraction_positive"]
                        for v in group
                    ]
                )
            ),
            "attack_vs_control_auroc": {
                kind: float(
                    np.mean(
                        [v["controls"][kind]["attack_vs_control_auroc"] for v in group]
                    )
                )
                for kind in controls
            },
        }
    result["removal"] = {}
    for layer in [20, 31]:
        z = np.array([r[f"z{layer}"] for r in project])
        norm = np.array([r[f"norm{layer}"] for r in project])
        before = np.array([r[f"z{layer}"] for r in plain])
        result["removal"][str(layer)] = {
            "projected_mean_abs_component": float(np.mean(abs(z))),
            "projected_max_abs_component": float(max(abs(z))),
            "projected_mean_abs_component_fraction": float(np.mean(abs(z) / norm)),
            "projected_max_abs_component_fraction": float(max(abs(z) / norm)),
            "plain_mean_abs_component": float(np.mean(abs(before))),
            "mean_abs_component_ratio_project_over_plain": float(
                np.mean(abs(z)) / np.mean(abs(before))
            ),
        }
    result["preferences"] = {
        arm: summarize_preferences(rs)
        for arm, rs in [("plain", plain), ("project", project)]
    }
    result["previous_score_reproduction"] = {}
    config = json.loads((out / "config.json").read_text())
    root = Path(__file__).resolve().parents[2]
    for arm, rs in [("plain", plain), ("project", project)]:
        old = read_rows(root / config["inputs"]["judge_" + arm]["path"])
        if [(r["id"], r["prompt_sha256"]) for r in rs] != [
            (r["id"], r["prompt_sha256"]) for r in old
        ]:
            raise ValueError("previous score identity differs")
        x = np.array([r["score"] for r in rs])
        y = np.array([r["score"] for r in old])
        result["previous_score_reproduction"][arm] = {
            "mae": float(np.mean(abs(x - y))),
            "correlation": float(np.corrcoef(x, y)[0, 1]),
            "decision_flips": int(np.sum((x >= 0.5) != (y >= 0.5))),
        }
    result["prepare"] = json.loads((out / "prepare_receipt.json").read_text())
    result["timing"] = {}
    for population in ["plain", "project", "natural", "whitespace"]:
        batches = [
            json.loads(p.read_text())
            for p in (out / "batches" / population).glob("*.json")
        ]
        rs = read_rows(out / (population + ".jsonl"))
        seconds = sum(b["seconds"] for b in batches)
        tokens = sum(b["tokens"] for b in batches)
        result["timing"][population] = {
            "rows": len(rs),
            "input_tokens": tokens,
            "seconds": seconds,
            "input_tokens_per_second": tokens / seconds,
            "requests_per_second": len(rs) / seconds,
            "latency_p50_p95_seconds": np.quantile(
                [r["latency_seconds"] for r in rs], [0.5, 0.95]
            ).tolist(),
            "latency_includes_semaphore_wait": True,
        }
    write_json(out / "summary.json", result)
