"""Metrics of the released-stack rescoring, next to the API baselines.

``PYTHONPATH=src:. python -m experiments.evilgenie_released_stack.analyze``
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.evilgenie_api_baselines import run as api

ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads(Path(__file__).with_name("config.json").read_text())
OUT = ROOT / CONFIG["results_dir"]


def ood_line(model: str) -> float:
    labels = {
        r["id"]: int(r["metadata"]["ground_truth"])
        for r in api.read_jsonl(ROOT / CONFIG["ood_prompts"])
    }
    scores = {
        r["id"]: float(r["score"])
        for r in api.read_jsonl(ROOT / CONFIG["models"][model]["ood_scores"])
    }
    if set(scores) != set(labels):
        raise ValueError(f"{model}: OOD coverage differs")
    return api.one_percent_line([s for i, s in scores.items() if labels[i] == 0])


def main() -> None:
    metas = {r["id"]: r["metadata"] for r in api.load("states.jsonl")}
    tokens = json.loads((api.DATA / "gleipnir_tokens.json").read_text())
    shared = api.gleipnir_shared_ids(metas, tokens)
    report: dict = {"monitors": {}}
    for model in CONFIG["models"]:
        rows = api.read_jsonl(OUT / f"{model}.jsonl")
        scores = {r["id"]: float(r["score"]) for r in rows}
        if set(scores) != set(metas):
            raise ValueError(f"{model}: {len(scores)} of {len(metas)} rows scored")
        line = ood_line(model)
        entry: dict = {
            "ood_threshold": line,
            "rows_over_training_length": int(
                sum(r["prompt_tokens"] > 29_696 for r in rows)
            ),
        }
        for name, ids in (("all", list(metas)), ("gleipnir_shared", sorted(shared))):
            ids = [
                i for i in ids if metas[i]["label_class"] in ("positive", "negative")
            ]
            y = np.array(
                [metas[i]["label_class"] == "positive" for i in ids], dtype=int
            )
            s = np.array([scores[i] for i in ids])
            p = np.array([metas[i]["problem_id"] for i in ids])
            entry[name] = api.metrics(y, s, p, line)
        clean = [
            scores[i] for i, m in metas.items() if m["label_class"] == "clean_negative"
        ]
        entry["flagged_clean_negative"] = [
            int(sum(c > line for c in clean)),
            len(clean),
        ]
        long_ids = [r["id"] for r in rows if r["prompt_tokens"] > 29_696]
        entry["over_training_length_classes"] = dict(
            Counter(metas[i]["label_class"] for i in long_ids)
        )
        report["monitors"][model] = entry
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
