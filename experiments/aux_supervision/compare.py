"""Apply the frozen ID advancement rule to the auxiliary-supervision arm."""

from __future__ import annotations

import argparse

from experiments.aux_supervision.prepare import ARM, ARMS, ROOT, receipt
from experiments.distillation_temperature.compare import CONTEXT, CONTROL, RULE, decide
from experiments.teacher_seed_variation.compare import load, score_difference, summary
from gleipnir.campaigns.monitoring.evaluation import attach_inputs
from gleipnir.data.monitoring import file_hash, read_rows, write_json


def compare(arm: str = ARM) -> dict:
    control_rows = load(CONTROL, "id")
    control = summary(control_rows)
    path = f"results/{arm}/evaluation/id.jsonl"
    # ID rows carry the arm's own rendered-prompt metadata.
    inputs = read_rows(ROOT / "data" / arm / "id_prompts.jsonl")
    rows = attach_inputs(read_rows(ROOT / path), inputs)
    metrics = summary(rows)
    report = {
        "rule": RULE,
        "control": control,
        "context": {k: summary(load(p, "id")) for k, p in CONTEXT.items()},
        "arm": {
            "campaign_id": arm,
            "metrics": metrics,
            "decision": decide(metrics, control),
            "scores_vs_control": score_difference(rows, control_rows),
            "sha256": file_hash(ROOT / path),
        },
    }
    print(arm, report["arm"]["decision"], flush=True)
    write_json(receipt(arm, "comparison.json"), report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, default=ARM)
    compare(parser.parse_args().arm)
