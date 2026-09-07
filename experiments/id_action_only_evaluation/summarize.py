"""Compare complete cleaned-ID predictions with the fixed original-ID baseline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.id_action_only_evaluation.prepare import CONFIG, JOB, ROOT
from experiments.tool_trajectory_monitoring.benchmark_distilled_ood import (
    validate_inputs,
)
from experiments.tool_trajectory_monitoring.benchmark_gpt_oss_ood import summarize
from experiments.tool_trajectory_monitoring.prepare_distillation_ood import (
    atomic_write_json,
    read_jsonl,
    sha256_file,
)
from gleipnir.evaluation_shards import merge_predictions


def main() -> None:
    config = json.loads(CONFIG.read_text())
    baseline_path = Path(config["baseline"]["predictions"])
    if sha256_file(baseline_path) != config["baseline"]["predictions_sha256"]:
        raise ValueError("Historical baseline predictions drift")
    rows = validate_inputs(config)
    path = ROOT / "id_evaluation/4b/adapters" / JOB / "predictions.jsonl"
    cleaned = merge_predictions(rows, [read_jsonl(path)], sha256_file(CONFIG))
    baseline = read_jsonl(baseline_path)
    original = {r["id"]: r for r in baseline}
    if len(original) != len(baseline) or set(original) != {r["id"] for r in cleaned}:
        raise ValueError("Baseline/cleaned identities differ")
    paired = {}
    for source in ["all", *config["scope"]["sources"]]:
        subset = [r for r in cleaned if source == "all" or r["source"] == source]
        for row in subset:
            before = original[row["id"]]
            if (row["source"], row["label"]) != (before["source"], before["label"]):
                raise ValueError("Paired source/label drift")
        a = np.array([original[r["id"]]["score"] for r in subset])
        b = np.array([r["score"] for r in subset])
        paired[source] = {
            "rows": len(subset),
            "mean_signed_score_change": float((b - a).mean()),
            "mean_absolute_score_change": float(np.abs(b - a).mean()),
            "pearson_correlation": float(np.corrcoef(a, b)[0, 1]),
            "threshold_flips": int(((a >= 0.5) != (b >= 0.5)).sum()),
            "positive_to_negative": int(((a >= 0.5) & (b < 0.5)).sum()),
            "negative_to_positive": int(((a < 0.5) & (b >= 0.5)).sum()),
        }
    old, new = summarize(baseline), summarize(cleaned)
    result = {
        "baseline": old,
        "cleaned": new,
        "paired": paired,
        "config_sha256": sha256_file(CONFIG),
        "baseline_predictions_sha256": sha256_file(baseline_path),
        "cleaned_predictions_sha256": sha256_file(path),
    }
    atomic_write_json(ROOT / "comparison.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
