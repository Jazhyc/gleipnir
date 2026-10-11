"""Bind one full-epoch variant to existing strict data and FP32-head controls."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from gleipnir.data.monitoring import file_hash, write_json

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
NAME = "strict-pauc02"
DATA = ROOT / "data" / NAME
OUT = ROOT / "results" / NAME


def prepare() -> None:
    if DATA.exists():
        raise ValueError("preserve existing prepared variant")
    config = copy.deepcopy(
        yaml.safe_load((ROOT / "data/fp32-monitor-head01/config.yaml").read_text())
    )
    spec = yaml.safe_load((HERE / "config.yaml").read_text())
    if spec["campaign_id"] != NAME or spec["promotion"]:
        raise ValueError("predeclared identity/promotion drift")
    config.update(
        campaign_id=NAME,
        teacher_filter="strict",
        hypothesis="One small partial-AUC ranking term improves strict-model "
        "ID ranking and OOD transfer.",
        intervention="Soft BCE plus weight 0.1 source-macro top-20%-negative "
        "pairwise softplus with logical-batch gradient replay.",
        selection="Only final one-epoch checkpoint; no weight sweep, "
        "early stopping or OOD tuning.",
        stop_condition="Stop on input/runtime drift, replay/finite-gradient "
        "failure, failed parity, missing coverage or three-hour stage deadline.",
        partial_auc=spec,
        baselines={},
    )
    config.pop("parent_campaign")
    config["matched_baselines"] = {
        "id": "results/fp32-monitor-head01",
        "ood": "results/fp32-monitor-head01",
    }
    config["extra_sources"] = [
        str(p.relative_to(ROOT))
        for directory in (
            HERE,
            ROOT / "experiments/monitor_score_resolution",
            ROOT / "experiments/teacher_disagreement_filtering",
        )
        for p in directory.iterdir()
        if p.is_file() and p.suffix in (".py", ".md", ".yaml")
    ]
    for name in ("id", "ood"):
        path = ROOT / "results/fp32-monitor-head01/evaluation" / f"{name}.jsonl"
        config["inputs"]["baseline_" + name] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": file_hash(path),
        }
    DATA.mkdir(parents=True)
    path = DATA / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    write_json(
        DATA / "design.json",
        {
            "hypothesis": config["hypothesis"],
            "partial_auc": spec,
            "baseline": "existing strict one-epoch seed-0 checkpoint, "
            "matched FP32 head",
            "qualification": "single seed; logical-batch tail approximation; "
            "extra score-pass compute",
            "evaluations": {"id": 3012, "ood": 6395},
            "selection": config["selection"],
        },
    )
    print("single_partial_auc_variant_prepared", NAME, flush=True)


if __name__ == "__main__":
    prepare()
