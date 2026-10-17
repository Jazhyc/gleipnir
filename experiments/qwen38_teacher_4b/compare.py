"""Apply the frozen ID advancement rule to the Qwen3.8-teacher arm."""

from __future__ import annotations

from experiments.distillation_temperature.compare import CONTEXT, CONTROL, RULE, decide
from experiments.qwen38_teacher_4b.prepare import ARM, ROOT
from experiments.teacher_seed_variation.compare import load, score_difference, summary
from gleipnir.data.monitoring import file_hash, write_json


def compare() -> dict:
    control_rows = load(CONTROL, "id")
    control = summary(control_rows)
    path = f"results/{ARM}/evaluation/id.jsonl"
    rows = load(path, "id")
    arm = summary(rows)
    report = {
        "rule": RULE,
        "control": control,
        "context": {k: summary(load(p, "id")) for k, p in CONTEXT.items()},
        "arm": {
            "metrics": arm,
            "decision": decide(arm, control),
            "scores_vs_control": score_difference(rows, control_rows),
            "sha256": file_hash(ROOT / path),
        },
    }
    write_json(ROOT / "results/qwen38_teacher_4b/comparison.json", report)
    print(report["arm"]["decision"], flush=True)
    return report


if __name__ == "__main__":
    compare()
