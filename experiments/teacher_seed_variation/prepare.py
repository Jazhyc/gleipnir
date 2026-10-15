"""Freeze the seed-1 SDPA teacher campaign and its matched scoring contexts."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from gleipnir.campaigns.monitoring.contract import Campaign
from gleipnir.data.monitoring import file_hash, write_json

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PARENT = "teacher-disagreement-delta000-sdpa04"
SEED1 = "teacher-seed-sdpa-s1-01"
OOD = {
    "ood": (
        "data/student_injection_awareness/regular/ood.jsonl",
        "42556d4f6ef6f7715b3b27f0b4d6f11ae83ef9c8499009489799032c2abbbf73",
    ),
    "ood_workload": ("data/teacher-disagreement-strict-ood01/workload.jsonl", None),
}
OOD_MANIFEST = "data/teacher-disagreement-strict-ood01/manifest.json"
SELECTION = "experiments/b200_inference_benchmark/serving_default.json"
TRAINING_PYTHON = "/tmp/gleipnir-training-runtime/venv/bin/python"
# Scoring contexts: (parent results directory, splits). Seed 0 is rescored in
# the same serving session as a drift control for its cached Oct-11/12 scores.
SCORING = {
    "teacher-seed-sdpa-s1-ood01": (f"results/{SEED1}", ["ood"]),
    "teacher-seed-sdpa-s0-rescore01": (f"results/{PARENT}", ["id", "ood"]),
}
FIELDS = {
    "hypothesis": (
        "Seed-to-seed variation of the strict-filtered BF16/SDPA teacher is small "
        "relative to the FP4/FA4 teacher gap on OOD and ID."
    ),
    "intervention": (
        "Retrain the seed-0 strict-filtered BF16/SDPA campaign with seed 1 only "
        "(data order and packing); initialization, data, targets and schedule fixed."
    ),
    "selection": (
        "Final one-epoch checkpoint only; descriptive seed comparison on ID and "
        "OOD under the precision-selection decision; no promotion."
    ),
    "stop_condition": (
        "Stop on input/runtime drift, nonfinite or missing gradients, failed "
        "master/merge/serving parity, incomplete scoring or completion."
    ),
}


def seed_config(parent: dict, gpu_uuid: str, selection_sha: str) -> dict:
    """Change only seed, identity, GPU, merge slot, serving hash and contract text."""
    config = copy.deepcopy(parent)
    config.update(
        campaign_id=SEED1,
        seed=1,
        gpu_uuid=gpu_uuid,
        merged_model=f"/tmp/gleipnir-merged/{SEED1}",
        extra_sources=sorted(
            str(p.relative_to(ROOT)) for p in HERE.iterdir() if p.suffix == ".py"
        ),
        **FIELDS,
    )
    config["inputs"]["serving_selection"]["sha256"] = selection_sha
    # The NC2 training runtime is restored beside the EU serving runtime, whose
    # FA4/MXFP8 overlays differ; FlashQLA files match the seed-0 receipt.
    config["training_python"] = TRAINING_PYTHON
    return config


def scoring_config(seed: dict, name: str) -> dict:
    parent, splits = SCORING[name]
    config = copy.deepcopy(seed)
    config.update(
        campaign_id=name,
        merged_model=f"/tmp/gleipnir-merged/{name}",
        parent_campaign=parent,
        score_splits=splits,
    )
    config["evaluation"]["populations"] = {
        s: {"id": 3012, "ood": 6395}[s] for s in splits
    }
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
    seed = seed_config(parent, gpu_uuid, selection_sha)
    written = {}
    for name, config in [
        (SEED1, seed),
        *[(n, scoring_config(seed, n)) for n in SCORING],
    ]:
        data = ROOT / "data" / name
        if data.exists():
            raise ValueError(f"refuse to overwrite {data}")
        data.mkdir(parents=True)
        (data / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        written[name] = str((data / "config.yaml").relative_to(ROOT))
    for name in SCORING:
        ctx = Campaign(
            ROOT,
            ROOT / written[name],
            yaml.safe_load((ROOT / written[name]).read_text()),
        )
        parent_dir = ROOT / SCORING[name][0]
        files = {
            written[name]: file_hash(ROOT / written[name]),
            **{spec["path"]: spec["sha256"] for spec in ctx.config["inputs"].values()},
        }
        write_json(
            ROOT / "data" / name / "manifest.json",
            {
                "files_sha256": files,
                "source_sha256": ctx.sources(),
                "parent": str(parent_dir.relative_to(ROOT)),
            },
        )
    report = {
        "status": "prepared",
        "gpu_uuid": gpu_uuid,
        "parent_config_sha256": file_hash(ROOT / "data" / PARENT / "config.yaml"),
        "serving_selection": {
            "parent_sha256": parent["inputs"]["serving_selection"]["sha256"],
            "current_sha256": selection_sha,
            "note": "BF16 scoring replaces the optimized workers; seed 0 is rescored.",
        },
        "configs": written,
    }
    write_json(ROOT / "results/teacher_seed_variation/preparation.json", report)
    print("seed_variation_prepared", json.dumps(written), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid", required=True)
    prepare(parser.parse_args().gpu_uuid)
