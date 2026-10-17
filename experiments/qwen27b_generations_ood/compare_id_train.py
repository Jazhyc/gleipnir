"""Compare unadapted Qwen3.8-27B with Kimi K3 targets (training rows) and the
trained 4B students (ID), by source-macro AUROC / pAUROC@20."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from gleipnir.evaluation.metrics import normalized_partial_auroc

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/qwen27b_generations_ood"
KIMI = ROOT / "data/tool_trajectory_monitoring/distillation_scaling/soft_targets.jsonl"
STRICT = ROOT / "data/teacher-disagreement-delta000-sdpa04/training.jsonl"
STUDENTS = {
    "4B SDPA seed 0": "results/teacher-seed-sdpa-s0-rescore01/evaluation/id.jsonl",
    "4B SDPA seed 1": "results/teacher-seed-sdpa-s1-01/evaluation/id.jsonl",
    "4B SDPA seed 2": "results/teacher-seed-sdpa-s2-01/evaluation/id.jsonl",
}


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def macro(records: list[tuple[str, int, float]]) -> dict:
    by = defaultdict(list)
    for source, label, score in records:
        by[source].append((label, score))
    groups = {}
    for source, values in sorted(by.items()):
        y = np.array([v[0] for v in values])
        s = np.array([v[1] for v in values])
        if len(set(y)) < 2:
            continue
        groups[source] = {
            "n": len(y),
            "auroc": float(roc_auc_score(y, s)),
            "pauroc_at_20": float(normalized_partial_auroc(y, s)),
            "accuracy": float(((s >= 0.5) == (y == 1)).mean()),
        }
    return {
        "macro_auroc": float(np.mean([g["auroc"] for g in groups.values()])),
        "macro_pauroc_at_20": float(
            np.mean([g["pauroc_at_20"] for g in groups.values()])
        ),
        "groups": groups,
    }


def main() -> None:
    report = {}
    # Training rows: Qwen3.8 vs. Kimi K3 on the same prompts.
    qwen = {r["id"]: r for r in rows(OUT / "qwen38_27b_train/predictions.jsonl")}
    prompts = {
        r["id"]: r
        for r in rows(
            ROOT / "data/tool_trajectory_monitoring/teacher_training/prompts.jsonl"
        )
    }
    # Some rows share an identical prompt; join by row identity, then check it.
    kimi = {str(r["index"]): r for r in rows(KIMI)}
    strict = {str(r["index"]) for r in rows(STRICT)}
    joined = []
    for rid, row in prompts.items():
        meta = row["metadata"]
        k = kimi[rid]
        if (
            k["rendered_prompt_sha256"] != meta["rendered_prompt_sha256"]
            or qwen[rid]["source_prompt_sha256"] != meta["rendered_prompt_sha256"]
        ):
            raise ValueError(f"prompt identity drift: {rid}")
        joined.append(
            (
                meta["source_dataset"],
                int(meta["ground_truth"]),
                float(qwen[rid]["score"]),
                float(k["soft_target"]),
                rid in strict,
            )
        )
    for subset, keep in {
        "all_8688": lambda r: True,
        "strict_filtered_7836": lambda r: r[4],
        "removed_852": lambda r: not r[4],
    }.items():
        chosen = [r for r in joined if keep(r)]
        q = np.array([r[2] for r in chosen])
        k = np.array([r[3] for r in chosen])
        y = np.array([r[1] for r in chosen])
        report[f"train/{subset}"] = {
            "rows": len(chosen),
            "qwen38": macro([(r[0], r[1], r[2]) for r in chosen]),
            "kimi_k3": macro([(r[0], r[1], r[3]) for r in chosen]),
            "qwen_kimi_score_correlation": float(np.corrcoef(q, k)[0, 1]),
            "qwen_kimi_side_agreement": float(((q >= 0.5) == (k >= 0.5)).mean()),
            "qwen_label_accuracy": float(((q >= 0.5) == (y == 1)).mean()),
            "kimi_label_accuracy": float(((k >= 0.5) == (y == 1)).mean()),
        }
    # ID: Qwen3.8 vs. the trained 4B students on the identical 3,012 rows.
    id_rows = {
        r["id"]: r
        for r in rows(
            ROOT / "data/tool_trajectory_monitoring/teacher_id_benchmark/prompts.jsonl"
        )
    }
    q_id = {r["id"]: r for r in rows(OUT / "qwen38_27b_id/predictions.jsonl")}
    report["id/qwen38"] = macro(
        [
            (
                m["metadata"]["source_dataset"],
                int(m["metadata"]["ground_truth"]),
                float(q_id[i]["score"]),
            )
            for i, m in id_rows.items()
        ]
    )
    for name, path in STUDENTS.items():
        report[f"id/{name}"] = macro(
            [
                (r["source_dataset"], int(r["ground_truth"]), float(r["score"]))
                for r in rows(ROOT / path)
            ]
        )
    (OUT / "id_train_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    for key, value in report.items():
        if key.startswith("train/"):
            q, k = value["qwen38"], value["kimi_k3"]
            print(
                f"{key} {value['rows']} "
                f"qwen38 pAUC {q['macro_pauroc_at_20']:.4f} "
                f"AUROC {q['macro_auroc']:.4f}"
                f" | kimi pAUC {k['macro_pauroc_at_20']:.4f} "
                f"AUROC {k['macro_auroc']:.4f}"
                f" | corr {value['qwen_kimi_score_correlation']:.3f}"
                f" agree {value['qwen_kimi_side_agreement']:.3f}"
                f" acc q {value['qwen_label_accuracy']:.3f}"
                f" k {value['kimi_label_accuracy']:.3f}"
            )
        else:
            groups = {
                s: round(g["pauroc_at_20"], 4) for s, g in value["groups"].items()
            }
            print(
                f"{key} pAUC {value['macro_pauroc_at_20']:.4f} "
                f"AUROC {value['macro_auroc']:.4f} {groups}"
            )


if __name__ == "__main__":
    main()
