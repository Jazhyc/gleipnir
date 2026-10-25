"""Freeze three hard-label-only 4B seeds matched to the soft-only release seeds.

Each seed copies the seed-0 strict-filtered BF16/SDPA campaign
(``teacher-disagreement-delta000-sdpa04``) and changes only the seed, the
objective (soft weight 0, direct source-label weight 1, so no Kimi artifact
reaches the trainer), identity, GPU, merge slot, serving hash and contract
text. Each training campaign scores ID itself; a separate context scores the
frozen OOD suite once, as for the soft seeds.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from experiments.teacher_seed_variation.prepare import (
    OOD,
    OOD_MANIFEST,
    PARENT,
    SELECTION,
    TRAINING_PYTHON,
)
from gleipnir.campaigns.monitoring.contract import Campaign
from gleipnir.data.monitoring import file_hash, write_json

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SEEDS = (0, 1, 2)
FIELDS = {
    "hypothesis": (
        "Kimi K3 soft targets give the 4B LoRA monitor higher OOD pAUROC@20 than "
        "source hard labels alone (matched seeds of the strict-filtered recipe)."
    ),
    "intervention": (
        "Retrain the seed-0 strict-filtered BF16/SDPA campaign with seed {seed} and "
        "source hard labels only (soft weight 0, direct weight 1); data, "
        "initialization, schedule and recipe fixed."
    ),
    "selection": (
        "Final one-epoch checkpoint only; descriptive 3-vs-3 comparison with the "
        "soft-only seeds on ID and the frozen OOD suite; no promotion."
    ),
    "stop_condition": (
        "Stop on input/runtime drift, nonfinite or missing gradients, failed "
        "master/merge/serving parity, incomplete scoring or completion."
    ),
}


def campaign(seed: int) -> str:
    return f"teacher-hard-sdpa-s{seed}-01"


def ood_context(seed: int) -> str:
    return f"teacher-hard-sdpa-s{seed}-ood01"


def seed_config(parent: dict, gpu_uuid: str, selection_sha: str, seed: int) -> dict:
    config = copy.deepcopy(parent)
    config.update(
        campaign_id=campaign(seed),
        seed=seed,
        soft_loss_weight=0.0,
        direct_loss_weight=1.0,
        gpu_uuid=gpu_uuid,
        merged_model=f"/tmp/gleipnir-merged/{campaign(seed)}",
        training_python=TRAINING_PYTHON,
        extra_sources=sorted(
            str(p.relative_to(ROOT))
            for p in HERE.iterdir()
            if p.suffix in (".py", ".md")
        ),
        **{k: v.format(seed=seed) for k, v in FIELDS.items()},
    )
    config["inputs"]["serving_selection"]["sha256"] = selection_sha
    return config


def scoring_config(seed: dict) -> dict:
    name = ood_context(seed["seed"])
    config = copy.deepcopy(seed)
    config.update(
        campaign_id=name,
        merged_model=f"/tmp/gleipnir-merged/{name}",
        parent_campaign=f"results/{campaign(seed['seed'])}",
        score_splits=["ood"],
        evaluations=[],
    )
    config["evaluation"]["populations"] = {"ood": 6395}
    for key, (path, _) in OOD.items():
        config["inputs"][key] = {"path": path, "sha256": file_hash(ROOT / path)}
    return config


def prepare(gpu_uuid: str, seeds: tuple[int, ...] = SEEDS) -> dict:
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
    written = {}
    for seed_number in seeds:
        seed = seed_config(parent, gpu_uuid, selection_sha, seed_number)
        for name, config in ((campaign(seed_number), seed), (None, None)):
            if name is None:
                name, config = ood_context(seed_number), scoring_config(seed)
            data = ROOT / "data" / name
            if data.exists():
                raise ValueError(f"refuse to overwrite {data}")
            data.mkdir(parents=True)
            (data / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
            written[name] = str((data / "config.yaml").relative_to(ROOT))
        name = ood_context(seed_number)
        ctx = Campaign(
            ROOT,
            ROOT / written[name],
            yaml.safe_load((ROOT / written[name]).read_text()),
        )
        files = {
            written[name]: file_hash(ROOT / written[name]),
            **{spec["path"]: spec["sha256"] for spec in ctx.config["inputs"].values()},
        }
        write_json(
            ROOT / "data" / name / "manifest.json",
            {
                "files_sha256": files,
                "source_sha256": ctx.sources(),
                "parent": f"results/{campaign(seed_number)}",
            },
        )
    report = {
        "status": "prepared",
        "gpu_uuid": gpu_uuid,
        "parent_config_sha256": file_hash(ROOT / "data" / PARENT / "config.yaml"),
        "serving_selection_sha256": selection_sha,
        "configs": written,
    }
    suffix = "" if tuple(seeds) == SEEDS else "_s" + "".join(map(str, seeds))
    write_json(ROOT / f"results/hard_label_seeds/preparation{suffix}.json", report)
    print("hard_label_seeds_prepared", json.dumps(written), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    prepare(args.gpu_uuid, tuple(args.seeds))
