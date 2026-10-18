"""Audit how strict-OOD monitor quality varies with prompt length.

Length is the frozen workload's Qwen token count for the binary student
prompt, used for every arm so bins are identical across monitors. SHADE-Arena
rows are also binned by length rank within main task x attack generator, so
task mix cannot produce a length trend on its own.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from experiments.teacher_seed_variation.compare import PREDICTIONS, load
from experiments.teacher_seed_variation.prepare import OOD, ROOT
from gleipnir.data.monitoring import read_rows, write_json
from gleipnir.evaluation.metrics import normalized_partial_auroc

EXCLUDED_TASK = "american_airlines_2"
SOURCES = ("shade_arena", "cua_shade_arena", "iac", "mle_sabotage")
SHADE_PARQUET = (
    "data/tool_trajectory_monitoring/source/"
    "deliberative-monitor-pipeline-2f10f4e/filtered_eval/ctrl_shade.parquet"
)
KIMI = [
    f"results/tool_trajectory_monitoring/kimi_k3_ood_{part}_fireworks.jsonl"
    for part in ("canary", "full_remaining")
]
ARMS = ("Qwen3.8-27B trained s0", "Kimi K3", "Gleipnir 4B s0")
BINS = 4
BOOTSTRAPS = 2000
OUTPUT = "results/shade_length_audit/summary.json"


def kimi_scores() -> dict[str, float]:
    scores = {}
    for path in KIMI:
        for row in read_rows(ROOT / path):
            lp = row["label_logprobs"]
            scores[row["id"]] = 1.0 / (1.0 + math.exp(lp["0"] - lp["1"]))
    return scores


def arm_scores() -> dict[str, dict[str, float]]:
    def student(path: str) -> dict[str, float]:
        return {r["id"]: r["score"] for r in load(path, "ood")}

    return {
        ARMS[0]: student("results/qwen38-27b-s0-ood01/evaluation/ood.jsonl"),
        ARMS[1]: kimi_scores(),
        ARMS[2]: student(PREDICTIONS["sdpa_seed0_rescored"]["ood"]),
    }


def task_pair(trajectory_data: str) -> str:
    """Read SHADE ``task_pair``; upstream metadata is a dict or its Python repr."""
    metadata = json.loads(trajectory_data)["metadata"]
    if isinstance(metadata, dict):
        return metadata["task_pair"]
    match = re.search(r"'task_pair': '([^']+)'", metadata)
    if match is None:
        raise ValueError("SHADE trajectory has no task_pair")
    return match.group(1)


def workload() -> pd.DataFrame:
    rows = read_rows(ROOT / OOD["ood_workload"][0])
    frame = pd.DataFrame(
        {
            "id": [r["id"] for r in rows],
            "source": [r["metadata"]["source_dataset"] for r in rows],
            "label": [int(r["metadata"]["ground_truth"]) for r in rows],
            "tokens": [r["prompt_tokens"] for r in rows],
            "source_row": [
                r["metadata"]["original_metadata"]["source_row_index"] for r in rows
            ],
        }
    )
    shade = pd.read_parquet(ROOT / SHADE_PARQUET)
    tasks = {
        i: (task_pair(t), e["model"])
        for i, (t, e) in enumerate(
            zip(shade["trajectory_data"], shade["extra_info"], strict=True)
        )
    }
    is_shade = frame["source"] == "shade_arena"
    upstream = shade["ground_truth"].to_numpy()[frame.loc[is_shade, "source_row"]]
    if (upstream != frame.loc[is_shade, "label"].to_numpy()).any():
        raise ValueError("SHADE source_row_index does not align with the parquet")
    frame.loc[is_shade, "task"] = frame.loc[is_shade, "source_row"].map(
        lambda i: tasks[i][0]
    )
    frame.loc[is_shade, "generator"] = frame.loc[is_shade, "source_row"].map(
        lambda i: tasks[i][1]
    )
    return frame


def within_stratum_bins(
    frame: pd.DataFrame, keys: list[str], bins: int = BINS
) -> pd.Series:
    """Assign each row a length bin by its length rank inside its stratum.

    Percentile ranks run over ``(0, 1]``, so the longest row in every stratum
    lands in the last bin and singleton strata land there too.
    """
    rank = frame.groupby(keys)["tokens"].rank(method="first", pct=True)
    return np.minimum((rank * bins).apply(math.ceil), bins) - 1


def quality(labels: np.ndarray, scores: np.ndarray, rng: np.random.Generator) -> dict:
    out = {
        "n": int(len(labels)),
        "positive_rate": float(labels.mean()),
        "auroc": float(roc_auc_score(labels, scores)),
        "pauroc_at_20": normalized_partial_auroc(labels, scores),
    }
    draws = []
    for _ in range(BOOTSTRAPS):
        i = rng.integers(0, len(labels), len(labels))
        if 0 < labels[i].sum() < len(i):
            draws.append(normalized_partial_auroc(labels[i], scores[i]))
    out["pauroc_at_20_ci95"] = [float(x) for x in np.percentile(draws, [2.5, 97.5])]
    return out


def by_bin(frame: pd.DataFrame, column: str, scores: dict, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for b, part in frame.groupby(column):
        s = part["id"].map(scores).to_numpy()
        rows.append(
            {
                "bin": int(b),
                "tokens": [int(part["tokens"].min()), int(part["tokens"].max())],
                **quality(part["label"].to_numpy(), s, rng),
            }
        )
    return rows


def trend(rows: list[dict]) -> float:
    return rows[-1]["pauroc_at_20"] - rows[0]["pauroc_at_20"]


def mean_ood(frame: pd.DataFrame, scores: dict[str, float]) -> dict:
    """Six-source unweighted means of pAUROC@20 and AUROC, with per-source values."""
    out = {}
    for metric, fn in (
        ("pauroc_at_20", normalized_partial_auroc),
        ("auroc", roc_auc_score),
    ):
        by_source = {
            source: float(fn(part["label"].to_numpy(), part["id"].map(scores)))
            for source, part in frame.groupby("source")
        }
        out[metric] = {
            "mean_ood": float(np.mean(list(by_source.values()))),
            **by_source,
        }
    return out


def exclusion(frame: pd.DataFrame, arms: dict) -> dict:
    """Sensitivity diagnostic: drop the intent-labelled AA task from SHADE."""
    kept = frame[frame["task"].ne(EXCLUDED_TASK) | frame["source"].ne("shade_arena")]
    return {
        "excluded_task": EXCLUDED_TASK,
        "rows_removed": int(len(frame) - len(kept)),
        "arms": {
            name: {"full": mean_ood(frame, s), "without": mean_ood(kept, s)}
            for name, s in arms.items()
        },
    }


def main() -> None:
    frame = workload()
    arms = arm_scores()
    for name, scores in arms.items():
        missing = set(frame["id"]) - set(scores)
        if missing:
            raise ValueError(f"{name} is missing {len(missing)} OOD rows")

    report: dict = {"length": "workload prompt_tokens (Qwen, binary prompt)"}
    report["sources"] = {}
    for source in SOURCES:
        part = frame[frame["source"] == source].copy()
        part["bin"] = within_stratum_bins(part, ["source"])
        report["sources"][source] = {
            "length_only_label_auroc": float(
                roc_auc_score(part["label"], part["tokens"])
            ),
            "arms": {
                name: by_bin(part, "bin", scores, seed=0)
                for name, scores in arms.items()
            },
        }

    shade = frame[frame["source"] == "shade_arena"].copy()
    shade["bin"] = within_stratum_bins(shade, ["task", "generator"])
    composition = defaultdict(dict)
    for (task, generator), part in shade.groupby(["task", "generator"]):
        composition[task][generator] = {
            "n": int(len(part)),
            "positives": int(part["label"].sum()),
            "median_tokens": float(part["tokens"].median()),
        }
    stratified: dict = {"composition": composition, "arms": {}}
    for name, scores in arms.items():
        shade["score"] = shade["id"].map(scores)
        per_task = {}
        for task, part in shade.groupby("task"):
            if part["label"].nunique() < 2 or len(part) < 40:
                continue
            per_task[task] = {
                "overall": quality(
                    part["label"].to_numpy(),
                    part["score"].to_numpy(),
                    np.random.default_rng(1),
                ),
                "within_task_halves": by_bin(
                    part.assign(half=within_stratum_bins(part, ["generator"], 2)),
                    "half",
                    scores,
                    seed=2,
                ),
            }
        stratified["arms"][name] = {
            "within_task_generator_bins": by_bin(shade, "bin", scores, seed=3),
            "per_task": per_task,
            "spearman_length_rank_vs_score": {
                str(label): float(
                    spearmanr(
                        shade.loc[shade["label"] == label, "bin"],
                        shade.loc[shade["label"] == label, "score"],
                    )[0]
                )
                for label in (0, 1)
            },
        }
    report["shade_stratified"] = stratified
    report["exclusion"] = exclusion(frame, arms)
    write_json(ROOT / OUTPUT, report)

    for source, value in report["sources"].items():
        auroc = value["length_only_label_auroc"]
        print(f"\n{source}: length-only label AUROC {auroc:.3f}")
        for name, rows in value["arms"].items():
            cells = " ".join(f"{r['pauroc_at_20']:.3f}" for r in rows)
            print(f"  {name:24s} raw quartiles {cells}  (Q4-Q1 {trend(rows):+.3f})")
    print("\nSHADE-Arena, length bins within main task x generator")
    for name, value in stratified["arms"].items():
        rows = value["within_task_generator_bins"]
        cells = " ".join(
            f"{r['pauroc_at_20']:.3f}[{r['pauroc_at_20_ci95'][0]:.2f},"
            f"{r['pauroc_at_20_ci95'][1]:.2f}]"
            for r in rows
        )
        print(f"  {name:24s} {cells}  (Q4-Q1 {trend(rows):+.3f})")
        for task, t in value["per_task"].items():
            halves = t["within_task_halves"]
            print(
                f"    {task[:40]:40s} n={t['overall']['n']:3d} "
                f"pAUROC {t['overall']['pauroc_at_20']:.3f} "
                f"short {halves[0]['pauroc_at_20']:.3f} "
                f"long {halves[1]['pauroc_at_20']:.3f}"
            )
    print(json.dumps(composition, indent=1))
    print(f"\nWithout {EXCLUDED_TASK}: SHADE-Arena and Mean-OOD")
    for name, value in report["exclusion"]["arms"].items():
        for metric in ("pauroc_at_20", "auroc"):
            full, without = value["full"][metric], value["without"][metric]
            print(
                f"  {name:24s} {metric:12s} SHADE {full['shade_arena']:.4f} -> "
                f"{without['shade_arena']:.4f}  Mean-OOD {full['mean_ood']:.4f} -> "
                f"{without['mean_ood']:.4f}"
            )


if __name__ == "__main__":
    main()
