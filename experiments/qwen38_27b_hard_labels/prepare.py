"""Freeze a hard-label-only Qwen3.8-27B campaign matched to the soft 27B seed 0.

Copies the first 27B campaign (``qwen38-27b-s0-01``) and changes only the
objective (soft weight 0, direct source-label weight 1, so no Kimi artifact
reaches the trainer), identity, GPU, merge slot, serving hash and contract
text. The campaign scores ID itself; a separate context scores the frozen OOD
suite, as for the soft 27B and the 4B hard seeds.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from experiments.qwen38_27b_campaign.prepare import ARM as PARENT
from experiments.teacher_seed_variation.prepare import OOD, OOD_MANIFEST, SELECTION
from gleipnir.campaigns.monitoring.contract import Campaign
from gleipnir.data.monitoring import file_hash, write_json
from gleipnir.repository import repo_root

ROOT = repo_root(__file__)
HERE = Path(__file__).resolve().parent
ARM = "qwen38-27b-hard-s0-01"
OOD_CONTEXT = "qwen38-27b-hard-s0-ood01"
FIELDS = {
    "hypothesis": (
        "Kimi K3 soft targets give the Qwen3.8-27B LoRA monitor higher ID and OOD "
        "pAUROC@20 than source hard labels alone (seed 0 of the strict-filtered "
        "recipe), as they do for the 4B."
    ),
    "intervention": (
        "Retrain the seed-0 Qwen3.8-27B campaign with source hard labels only "
        "(soft weight 0, direct weight 1); data, initialization, schedule and "
        "recipe fixed."
    ),
    "selection": (
        "Final one-epoch checkpoint only; descriptive comparison with the soft "
        "27B seed 0, the unadapted 27B and the 4B hard/soft seeds on ID and the "
        "frozen OOD suite; no promotion."
    ),
    "stop_condition": (
        "Stop on input/runtime drift, failed startup gates, nonfinite or missing "
        "gradients, OOM, failed master/merge/serving parity, incomplete scoring "
        "or completion."
    ),
}


def arm_config(parent: dict, gpu_uuid: str, selection_sha: str) -> dict:
    config = copy.deepcopy(parent)
    config.update(
        campaign_id=ARM,
        soft_loss_weight=0.0,
        direct_loss_weight=1.0,
        gpu_uuid=gpu_uuid,
        merged_model=f"/tmp/gleipnir-merged/{ARM}",
        extra_sources=sorted(
            str(p.relative_to(ROOT))
            for p in HERE.iterdir()
            if p.suffix in (".py", ".md")
        ),
        **FIELDS,
    )
    config["inputs"]["serving_selection"]["sha256"] = selection_sha
    return config


def scoring_config(arm: dict) -> dict:
    config = copy.deepcopy(arm)
    config.update(
        campaign_id=OOD_CONTEXT,
        merged_model=f"/tmp/gleipnir-merged/{OOD_CONTEXT}",
        parent_campaign=f"results/{ARM}",
        score_splits=["ood"],
        evaluations=[],
    )
    config["evaluation"]["populations"] = {"ood": 6395}
    for key, (path, _) in OOD.items():
        config["inputs"][key] = {"path": path, "sha256": file_hash(ROOT / path)}
    return config


def prepare(gpu_uuid: str) -> dict:
    if not gpu_uuid.startswith("GPU-"):
        raise ValueError("a verified live GPU UUID is required")
    parent = yaml.safe_load((ROOT / "data" / PARENT / "config.yaml").read_text())
    for name, spec in parent["inputs"].items():
        if (
            name != "serving_selection"
            and file_hash(ROOT / spec["path"]) != spec["sha256"]
        ):
            raise ValueError(f"parent input drift: {name}")
    for key, (path, expected) in OOD.items():
        if expected and file_hash(ROOT / path) != expected:
            raise ValueError(f"OOD input drift: {key}")
    frozen = json.loads((ROOT / OOD_MANIFEST).read_text())["files_sha256"]
    if frozen[OOD["ood_workload"][0]] != file_hash(ROOT / OOD["ood_workload"][0]):
        raise ValueError("frozen OOD workload drift")
    selection_sha = file_hash(ROOT / SELECTION)
    arm = arm_config(parent, gpu_uuid, selection_sha)
    written = {}
    for name, config in ((ARM, arm), (OOD_CONTEXT, scoring_config(arm))):
        data = ROOT / "data" / name
        if data.exists():
            raise ValueError(f"refuse to overwrite {data}")
        data.mkdir(parents=True)
        (data / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        written[name] = str((data / "config.yaml").relative_to(ROOT))
    ctx = Campaign(
        ROOT,
        ROOT / written[OOD_CONTEXT],
        yaml.safe_load((ROOT / written[OOD_CONTEXT]).read_text()),
    )
    write_json(
        ROOT / "data" / OOD_CONTEXT / "manifest.json",
        {
            "files_sha256": {
                written[OOD_CONTEXT]: file_hash(ROOT / written[OOD_CONTEXT]),
                **{s["path"]: s["sha256"] for s in ctx.config["inputs"].values()},
            },
            "source_sha256": ctx.sources(),
            "parent": f"results/{ARM}",
        },
    )
    report = {
        "status": "prepared",
        "gpu_uuid": gpu_uuid,
        "parent_config_sha256": file_hash(ROOT / "data" / PARENT / "config.yaml"),
        "serving_selection_sha256": selection_sha,
        "configs": written,
    }
    write_json(ROOT / "results/qwen38_27b_hard_labels/preparation.json", report)
    print("qwen38_27b_hard_labels_prepared", json.dumps(written), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid", required=True)
    prepare(parser.parse_args().gpu_uuid)
