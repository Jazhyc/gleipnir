"""Apply the frozen ID advancement rule to the auxiliary-supervision arm."""

from __future__ import annotations

from experiments.aux_supervision.prepare import ARM, OUT, ROOT
from experiments.distillation_temperature.compare import CONTEXT, CONTROL, RULE, decide
from experiments.teacher_seed_variation.compare import load, score_difference, summary
from gleipnir.campaigns.monitoring.evaluation import attach_inputs
from gleipnir.data.monitoring import file_hash, read_rows, write_json


def compare() -> dict:
    control_rows = load(CONTROL, "id")
    control = summary(control_rows)
    path = f"results/{ARM}/evaluation/id.jsonl"
    # ID rows carry the arm's own rendered-prompt metadata.
    inputs = read_rows(ROOT / "data" / ARM / "id_prompts.jsonl")
    rows = attach_inputs(read_rows(ROOT / path), inputs)
    metrics = summary(rows)
    report = {
        "rule": RULE,
        "control": control,
        "context": {k: summary(load(p, "id")) for k, p in CONTEXT.items()},
        "arm": {
            "campaign_id": ARM,
            "metrics": metrics,
            "decision": decide(metrics, control),
            "scores_vs_control": score_difference(rows, control_rows),
            "sha256": file_hash(ROOT / path),
        },
    }
    print(ARM, report["arm"]["decision"], flush=True)
    write_json(OUT / "comparison.json", report)
    return report


if __name__ == "__main__":
    compare()
