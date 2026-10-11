"""Freeze matched ID/OOD inputs and immutable strict-model BF16 baselines."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml

from gleipnir.campaigns.monitoring.contract import Campaign
from gleipnir.data.monitoring import file_hash, write_json

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
NAME = "fp32-monitor-head01"
DATA = ROOT / "data" / NAME
OUT = ROOT / "results" / NAME
PARENT = "results/teacher-disagreement-delta000-sdpa04"
OOD = "results/teacher-disagreement-strict-ood01"


def prepare() -> None:
    if DATA.exists():
        raise ValueError("refuse to overwrite prepared head experiment")
    old_data = ROOT / "data/teacher-disagreement-strict-ood01"
    spec = yaml.safe_load((HERE / "config.yaml").read_text())
    if spec["campaign_id"] != NAME or spec["promote"]:
        raise ValueError("frozen head experiment identity/promotion drift")
    config = copy.deepcopy(yaml.safe_load((old_data / "config.yaml").read_text()))
    files = json.loads((old_data / "manifest.json").read_text())["files_sha256"]
    config.update(
        campaign_id=NAME,
        evaluations=["id"],
        baselines={},
        hypothesis="FP32 final head recovers useful ordering within BF16 score ties.",
        intervention="Reuse identical strict BF16 merge/backbone; "
        "final projection only in FP32.",
        selection="Fixed strict final checkpoint, "
        "no ID/OOD tuning or recipe promotion.",
        stop_condition="Stop on drift, dtype/parity failure, "
        "missing scores or one-hour deadline.",
    )
    config["evaluation"]["populations"] = spec["populations"]
    config["evaluation"]["head_compute_precision"] = spec["head_compute_precision"]
    config["extra_sources"] = [
        str(p.relative_to(ROOT))
        for p in HERE.iterdir()
        if p.is_file() and p.suffix in (".py", ".md", ".yaml")
    ]
    config["matched_baselines"] = {"id": PARENT, "ood": OOD}
    for directory in (PARENT, OOD):
        for filename in (
            "merged_artifact.json",
            "merged_parity.json",
            "optimized_parity.json",
            "bf16_audit.json",
            "evaluation/complete.json",
            "evaluation/id.jsonl" if directory == PARENT else "evaluation/ood.jsonl",
            "evaluation/id_timing.json"
            if directory == PARENT
            else "evaluation/ood_timing.json",
        ):
            path = ROOT / directory / filename
            files[str(path.relative_to(ROOT))] = file_hash(path)
    DATA.mkdir(parents=True)
    path = DATA / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    files[str(path.relative_to(ROOT))] = file_hash(path)
    write_json(
        DATA / "manifest.json",
        {
            "files_sha256": files,
            "source_sha256": Campaign(ROOT, path, config).sources(),
            "selection": config["selection"],
            "populations": {"id": 3012, "ood": 6395},
            "precision": "BF16 backbone and stored weights; "
            "FP32 two-row computation/output",
        },
    )
    print("fp32_head_contract_prepared", len(files), flush=True)


if __name__ == "__main__":
    prepare()
