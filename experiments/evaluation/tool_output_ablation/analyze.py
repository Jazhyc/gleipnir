"""Compare stripped and full OOD inputs for the released Gleipnir monitors.

Per model and source: pAUROC@20 and recall at 1% FPR (threshold from the
benign rows of the same source and input), for the recorded full-input rows
(the baseline) and the stripped run. Mean-OOD averages the six sources with
equal weight; a second mean leaves out CUA-SHADE-Arena, whose rendering keeps
only tool names once outputs are removed.

``python -m experiments.evaluation.tool_output_ablation.analyze --name run01``
"""

from __future__ import annotations

import argparse
import json

from experiments.evaluation.ood_recall_at_fpr.compute import (
    evaluate,
    load_reference,
    load_scores,
)
from experiments.evaluation.tool_output_ablation.count import CONFIG, ROOT

RECALL_CONFIG = json.loads(
    (ROOT / "experiments/evaluation/ood_recall_at_fpr/config.json").read_text()
)
FLOOR_SOURCE = "cua_shade_arena"


def means(metrics: dict, key: str) -> dict[str, float]:
    sources = metrics["per_source"]
    kept = [s for s in sources if s != FLOOR_SOURCE]
    return {
        "all": sum(sources[s][key] for s in sources) / len(sources),
        "without_cua": sum(sources[s][key] for s in kept) / len(kept),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    reference = load_reference(RECALL_CONFIG)
    out_dir = ROOT / CONFIG["results_dir"] / args.name
    report: dict = {"fprs": CONFIG["fprs"], "models": {}}
    for model, spec in CONFIG["models"].items():
        inputs = {
            "full": spec["baseline"],
            "stripped": str((out_dir / f"{model}_ood.jsonl").relative_to(ROOT)),
        }
        entry: dict = {}
        for view, path in inputs.items():
            scores = load_scores(
                {"name": f"{model} {view}", "files": [path], "score_field": "score"},
                reference,
            )
            entry[view] = {}
            for fpr in CONFIG["fprs"]:
                metrics = evaluate(scores, reference, fpr)
                entry[view][str(fpr)] = {
                    "pauroc20": means(metrics, "pauroc20"),
                    "recall": means(metrics, "recall"),
                    "per_source": {
                        s: {k: v[k] for k in ("pauroc20", "recall", "achieved_fpr")}
                        for s, v in metrics["per_source"].items()
                    },
                }
        report["models"][model] = entry
        for fpr in CONFIG["fprs"]:
            full, stripped = entry["full"][str(fpr)], entry["stripped"][str(fpr)]
            print(f"== {model} (recall at {fpr:.0%} FPR)")
            for source in full["per_source"]:
                a, b = full["per_source"][source], stripped["per_source"][source]
                print(
                    f"  {source:16} pAUROC@20 {a['pauroc20']:.3f}"
                    f" -> {b['pauroc20']:.3f}"
                    f"  recall {a['recall']:.3f} -> {b['recall']:.3f}"
                )
            for subset in ("all", "without_cua"):
                print(
                    f"  mean {subset:11} pAUROC@20 {full['pauroc20'][subset]:.3f} ->"
                    f" {stripped['pauroc20'][subset]:.3f}  recall"
                    f" {full['recall'][subset]:.3f} -> {stripped['recall'][subset]:.3f}"
                )
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
