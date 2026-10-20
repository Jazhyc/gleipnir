"""Freeze the three-seed weight-average campaign and its OOD scoring context.

The campaign is the seed-1 control's (`teacher-seed-sdpa-s1-01`) with only its
identity changed; its "train" stage is replaced by the exact average of the
three BF16/SDPA seeds' FP32 masters (`average.py`). Merge, master/merged parity
and ID scoring run through the shared campaign stages; OOD is scored in a
separate context that re-merges and must reproduce the campaign's merge.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from experiments.teacher_seed_variation.prepare import OOD, OOD_MANIFEST
from gleipnir.campaigns.monitoring.contract import Campaign
from gleipnir.data.monitoring import file_hash, write_json

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "results/seed_average"
CONTROL = "teacher-seed-sdpa-s1-01"
AVERAGE = "teacher-seed-sdpa-avg012-01"
AVERAGE_OOD = "teacher-seed-sdpa-avg012-ood01"
# Same-hardware drift control: seed 1 rescored on ID on the B300.
SEED1_RESCORE = "teacher-seed-sdpa-s1-b300-rescore01"
SEEDS = {
    0: "teacher-disagreement-delta000-sdpa04",
    1: "teacher-seed-sdpa-s1-01",
    2: "teacher-seed-sdpa-s2-01",
}
SELECTION = "experiments/b200_inference_benchmark/serving_default.json"
FIELDS = {
    "hypothesis": (
        "The exact weight average of the three strict-filtered BF16/SDPA seeds "
        "(same initialization, different data order) ranks held-out ID at least "
        "as well as the seeds' mean, removing seed noise from the release."
    ),
    "intervention": (
        "No training: the seeds' FP32 LoRA updates are averaged exactly as one "
        "rank-384 LoRA (A stacked, B/3, alpha 768); data, prompt, base and "
        "serving are the seed-1 control's."
    ),
    "selection": (
        "Frozen before scoring: the average is the Gleipnir 4B release candidate "
        "if its ID macro pAUROC@20 is at least the three-seed mean; otherwise "
        "the ID-best seed is. OOD is scored once and reported, never selected on."
    ),
    "stop_condition": (
        "Stop on source-master drift, an inexact average, failed master/merge/"
        "serving parity, incomplete scoring or completion."
    ),
}


def masters() -> dict[str, dict]:
    out = {}
    for seed, campaign in SEEDS.items():
        adapter = ROOT / "results" / campaign / "4b/monitor"
        complete = json.loads((adapter / "complete.json").read_text())
        weights = adapter / "causal_adapter/adapter_model.safetensors"
        if file_hash(weights) != complete["master_sha256"]:
            raise ValueError(f"seed {seed} FP32 master drift")
        out[str(seed)] = {
            "campaign": campaign,
            "path": str(weights.relative_to(ROOT)),
            "sha256": complete["master_sha256"],
        }
    return out


def scoring_config(average: dict) -> dict:
    config = copy.deepcopy(average)
    config.update(
        campaign_id=AVERAGE_OOD,
        merged_model=f"/tmp/gleipnir-merged/{AVERAGE_OOD}",
        parent_campaign=f"results/{AVERAGE}",
        score_splits=["ood"],
    )
    config["evaluations"] = []
    config["evaluation"]["populations"] = {"ood": 6395}
    for key, (path, _) in OOD.items():
        config["inputs"][key] = {"path": path, "sha256": file_hash(ROOT / path)}
    return config


def rescore_config(gpu_uuid: str) -> dict:
    """Seed 1's own config, scoring only ID from its existing FP32 master."""
    config = yaml.safe_load((ROOT / "data" / CONTROL / "config.yaml").read_text())
    config.update(
        campaign_id=SEED1_RESCORE,
        gpu_uuid=gpu_uuid,
        merged_model=f"/tmp/gleipnir-merged/{SEED1_RESCORE}",
        parent_campaign=f"results/{CONTROL}",
        score_splits=["id"],
    )
    config["evaluations"] = ["id"]
    config["evaluation"]["populations"] = {"id": 3012}
    return config


def prepare_rescore(gpu_uuid: str) -> dict:
    data = ROOT / "data" / SEED1_RESCORE
    if data.exists():
        raise ValueError(f"refuse to overwrite {data}")
    data.mkdir(parents=True)
    path = data / "config.yaml"
    path.write_text(yaml.safe_dump(rescore_config(gpu_uuid), sort_keys=False))
    ctx = Campaign(ROOT, path, yaml.safe_load(path.read_text()))
    files = {
        str(path.relative_to(ROOT)): file_hash(path),
        **{spec["path"]: spec["sha256"] for spec in ctx.config["inputs"].values()},
    }
    manifest = {
        "files_sha256": files,
        "source_sha256": ctx.sources(),
        "parent": f"results/{CONTROL}",
    }
    write_json(data / "manifest.json", manifest)
    print("seed_average_rescore_prepared", SEED1_RESCORE, flush=True)
    return manifest


def prepare(gpu_uuid: str) -> dict:
    if not gpu_uuid.startswith("GPU-"):
        raise ValueError("a verified live GPU UUID is required")
    control = yaml.safe_load((ROOT / "data" / CONTROL / "config.yaml").read_text())
    for key, (path, expected) in OOD.items():
        if expected and file_hash(ROOT / path) != expected:
            raise ValueError(f"OOD input drift: {key}")
    frozen = json.loads((ROOT / OOD_MANIFEST).read_text())["files_sha256"]
    if frozen[OOD["ood_workload"][0]] != file_hash(ROOT / OOD["ood_workload"][0]):
        raise ValueError("frozen OOD workload drift")
    sources = masters()
    config = copy.deepcopy(control)
    config.update(
        campaign_id=AVERAGE,
        gpu_uuid=gpu_uuid,
        merged_model=f"/tmp/gleipnir-merged/{AVERAGE}",
        # The OOD stage runs the seed-variation scorer, so bind its sources too.
        extra_sources=sorted(
            str(p.relative_to(ROOT))
            for directory in (HERE, ROOT / "experiments/teacher_seed_variation")
            for p in directory.iterdir()
            if p.suffix == ".py"
        ),
        seed_average_sources=sources,
        **FIELDS,
    )
    config["inputs"]["serving_selection"]["sha256"] = file_hash(ROOT / SELECTION)
    written = {}
    for name, body in ((AVERAGE, config), (AVERAGE_OOD, scoring_config(config))):
        data = ROOT / "data" / name
        if data.exists():
            raise ValueError(f"refuse to overwrite {data}")
        data.mkdir(parents=True)
        (data / "config.yaml").write_text(yaml.safe_dump(body, sort_keys=False))
        written[name] = str((data / "config.yaml").relative_to(ROOT))
    path = ROOT / written[AVERAGE_OOD]
    ctx = Campaign(ROOT, path, yaml.safe_load(path.read_text()))
    files = {
        written[AVERAGE_OOD]: file_hash(ROOT / written[AVERAGE_OOD]),
        **{spec["path"]: spec["sha256"] for spec in ctx.config["inputs"].values()},
    }
    write_json(
        ROOT / "data" / AVERAGE_OOD / "manifest.json",
        {
            "files_sha256": files,
            "source_sha256": ctx.sources(),
            "parent": f"results/{AVERAGE}",
        },
    )
    report = {
        "status": "prepared",
        "gpu_uuid": gpu_uuid,
        "configs": written,
        "sources": sources,
    }
    write_json(OUT / "preparation.json", report)
    print("seed_average_prepared", json.dumps(written), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--rescore", action="store_true")
    args = parser.parse_args()
    (prepare_rescore if args.rescore else prepare)(args.gpu_uuid)
