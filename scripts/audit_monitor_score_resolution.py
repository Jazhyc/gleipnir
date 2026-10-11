"""Audit saved OOD score precision without new model inference or label tuning."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.teacher_disagreement_ood.prepare import (
    DATA,
    OUT,
    REFERENCES,
    SOURCE,
    bind_cached,
)
from gleipnir.campaigns.monitoring.evaluation import attach_inputs, id_metrics
from gleipnir.data.monitoring import file_hash, read_rows, write_json
from gleipnir.serving.monitor_score import score_payload, validate_score_response

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    manifest = json.loads((DATA / "manifest.json").read_text())
    for path in [SOURCE, *REFERENCES.values()]:
        if file_hash(ROOT / path) != manifest["files_sha256"][path]:
            raise ValueError(f"frozen input drift: {path}")
    canonical = read_rows(ROOT / SOURCE)
    predictions = OUT / "evaluation/ood.jsonl"
    done = json.loads((OUT / "evaluation/complete.json").read_text())
    if file_hash(predictions) != done["files_sha256"]["ood"]:
        raise ValueError("strict predictions changed")
    rows = read_rows(predictions)
    for row in rows:
        validate_score_response(row, row["prompt_tokens"])
    populations = {"strict": attach_inputs(rows, canonical)}
    populations.update(
        {
            name: bind_cached(read_rows(ROOT / path), canonical)
            for name, path in REFERENCES.items()
        }
    )
    report = {
        "scope": "saved-score rounding diagnostic, not FP32-head inference",
        "inputs_sha256": {
            path: file_hash(ROOT / path) for path in [SOURCE, *REFERENCES.values()]
        },
        "strict_predictions_sha256": file_hash(predictions),
        "source_sha256": file_hash(Path(__file__)),
        "cases": {},
    }
    for name, population in populations.items():
        if len(population) != 6395:
            raise ValueError("OOD coverage changed")
        margins = np.array(
            [
                r["margin"] if name == "strict" else r["logprob_1"] - r["logprob_0"]
                for r in population
            ]
        )
        rounded = np.round(margins * 8) / 8
        residual = float(np.max(np.abs(margins - rounded)))
        if residual > 3e-7:
            raise ValueError("margins are not effectively on the observed 0.125 grid")
        coarse = [
            {**row, "score": score_payload([0.0, float(margin)], 1)["score"]}
            for row, margin in zip(population, rounded, strict=True)
        ]
        original_metrics = id_metrics(population)
        coarse_metrics = id_metrics(coarse)
        original = original_metrics["metrics"]["macro"]["macro"]
        coarsened = coarse_metrics["metrics"]["macro"]["macro"]
        case = {
            "rows": len(population),
            "unique_scores_original": len({r["score"] for r in population}),
            "unique_margins_at_6_decimals": len(set(np.round(margins, 6))),
            "unique_grid_margins": len(set(rounded)),
            "maximum_margin_distance_to_grid": residual,
            "maximum_score_change": max(
                abs(r["score"] - c["score"])
                for r, c in zip(population, coarse, strict=True)
            ),
            "original_metrics": original_metrics,
            "coarsened_metrics": coarse_metrics,
            "source_macro_difference": {
                k: coarsened[k] - original[k] for k in original
            },
        }
        report["cases"][name] = case
        print(
            name,
            case["unique_scores_original"],
            case["unique_grid_margins"],
            case["source_macro_difference"],
            flush=True,
        )
    logits = np.array([r["logits"] for r in rows])
    report["strict_logits"] = {
        "minimum": float(logits.min()),
        "maximum": float(logits.max()),
        "all_on_0125_grid": bool(np.all(logits * 8 == np.round(logits * 8))),
    }
    write_json(OUT / "score_resolution_audit.json", report)


if __name__ == "__main__":
    main()
