"""Run the single regular-data CAFT campaign and its disjoint APPS analysis."""

import json
from pathlib import Path

from experiments.activation_filter_training.analyze import heldout_apps
from gleipnir.campaigns.monitoring.__main__ import execute
from gleipnir.campaigns.monitoring.contract import Campaign
from gleipnir.data.monitoring import read_rows, write_json

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


if __name__ == "__main__":
    ctx = Campaign.load(ROOT, HERE / "config.yaml")
    execute(ctx)
    partition = json.loads(ctx.input("direction_partition").read_text())
    fit = set(partition["fit_tasks"])
    reports = {
        "caft": heldout_apps(
            read_rows(ctx.output / "evaluation/benchmark.jsonl"),
            read_rows(ctx.output / "evaluation/honest_controls.jsonl"),
            fit,
        ),
        "cached_regular": heldout_apps(
            read_rows(ctx.input("baseline_benchmark")),
            read_rows(ctx.input("baseline_controls")),
            fit,
        ),
    }
    write_json(
        ROOT / "results/concept_ablation_training/heldout_apps.json",
        {
            "metrics": reports,
            ("qualification"): (
                "Same APPS holdout; user requests cached controls, whose prec"
                "ision/serving/host differ. CAFT uses regular data and unabla"
                "ted inference."
            ),
        },
    )
    print("caft_regular_complete", flush=True)
