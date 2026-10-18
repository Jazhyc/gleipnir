"""Apply the frozen ID advancement rule to both Gleipnir-teacher arms."""

from __future__ import annotations

from experiments.distillation_temperature.compare import CONTEXT, CONTROL, RULE, decide
from experiments.gleipnir27b_teacher_4b.prepare import ARMS, OUT, ROOT
from experiments.teacher_seed_variation.compare import load, score_difference, summary
from gleipnir.campaigns.monitoring.evaluation import attach_inputs
from gleipnir.data.monitoring import file_hash, read_rows, write_json


def arm_rows(arm: str) -> tuple[str, list[dict]]:
    path = f"results/{ARMS[arm]}/evaluation/id.jsonl"
    if arm == "student":
        return path, load(path, "id")
    # Teacher-prompt ID rows carry their own rendered-prompt metadata.
    inputs = read_rows(ROOT / "data" / ARMS[arm] / "id_prompts.jsonl")
    return path, attach_inputs(read_rows(ROOT / path), inputs)


def compare() -> dict:
    control_rows = load(CONTROL, "id")
    control = summary(control_rows)
    report = {
        "rule": RULE,
        "control": control,
        "context": {k: summary(load(p, "id")) for k, p in CONTEXT.items()},
        "arms": {},
    }
    for arm in ARMS:
        path, rows = arm_rows(arm)
        metrics = summary(rows)
        report["arms"][arm] = {
            "metrics": metrics,
            "decision": decide(metrics, control),
            "scores_vs_control": score_difference(rows, control_rows),
            "sha256": file_hash(ROOT / path),
        }
        print(arm, report["arms"][arm]["decision"], flush=True)
    write_json(OUT / "comparison.json", report)
    return report


if __name__ == "__main__":
    compare()
