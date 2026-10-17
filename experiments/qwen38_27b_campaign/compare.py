"""Compare the trained 27B on ID with the unadapted 27B and the 4B seeds."""

from __future__ import annotations

import json

import numpy as np

from experiments.qwen27b_generations_ood.compare_id_train import (
    OUT,
    ROOT,
    STUDENTS,
    macro,
    rows,
)

TRAINED = "results/qwen38-27b-s0-01/evaluation/id.jsonl"
ID_PROMPTS = ROOT / "data/tool_trajectory_monitoring/teacher_id_benchmark/prompts.jsonl"


def summary(records: list[tuple[str, int, float]]) -> dict:
    report = macro(records)
    y = np.array([r[1] for r in records])
    s = np.array([r[2] for r in records])
    report.update(rows=len(records), brier=float(((s - y) ** 2).mean()))
    return report


def main() -> None:
    def campaign(path: str) -> list[tuple[str, int, float]]:
        return [
            (r["source_dataset"], int(r["ground_truth"]), float(r["score"]))
            for r in rows(ROOT / path)
        ]

    zero = {r["id"]: r for r in rows(OUT / "qwen38_27b_id/predictions.jsonl")}
    arms = {
        "27B trained, seed 0": campaign(TRAINED),
        "27B unadapted": [
            (
                m["metadata"]["source_dataset"],
                int(m["metadata"]["ground_truth"]),
                float(zero[m["id"]]["score"]),
            )
            for m in rows(ID_PROMPTS)
        ],
        **{name: campaign(path) for name, path in STUDENTS.items()},
    }
    report = {name: summary(records) for name, records in arms.items()}
    (ROOT / "results/qwen38-27b-s0-01/id_comparison.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    for name, value in report.items():
        groups = {s: round(g["pauroc_at_20"], 4) for s, g in value["groups"].items()}
        print(
            f"{name}: n={value['rows']} pAUC {value['macro_pauroc_at_20']:.4f} "
            f"AUROC {value['macro_auroc']:.4f} Brier {value['brier']:.4f} {groups}"
        )


if __name__ == "__main__":
    main()
