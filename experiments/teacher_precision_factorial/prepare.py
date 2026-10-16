"""Freeze off-diagonal MLP-precision x attention teacher cells (seed 0)."""

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
# Each cell changes only the training profile and its startup reference.
CELLS = {
    "bf16-fa4": {
        "profile": "qwen35_4b_b200_bf16_fa4",
        "startup_reference": (
            "results/b200_bf16_fa4_accepted/flash_attention_4/causal_adapter/"
            "training_metadata.json"
        ),
    },
    # Composite validation: SDPA receipt for attention, FP4 receipt for MLP.
    "fp4-sdpa": {
        "profile": "qwen35_4b_b200_fp4_mlp_sdpa",
        "startup_reference": (
            "results/student_injection_awareness/4b/regular/causal_adapter/"
            "training_metadata.json"
        ),
    },
}
FIELDS = {
    "hypothesis": (
        "Attention implementation and MLP precision contribute separately to the "
        "FP4/FA4 teacher's OOD loss relative to BF16/SDPA."
    ),
    "intervention": (
        "Train the strict-filtered seed-0 teacher with the {cell} profile; data, "
        "targets, initialization, seed, optimizer and schedule fixed."
    ),
    "selection": (
        "Final one-epoch checkpoint; descriptive comparison against the three "
        "BF16/SDPA seeds and FP4/FA4 on OOD and ID; no promotion."
    ),
    "stop_condition": (
        "Stop on input/runtime drift, nonfinite or missing gradients, failed "
        "master/merge/serving parity, incomplete scoring or completion."
    ),
}


def campaign(cell: str) -> str:
    return f"teacher-cell-{cell}-s0-01"


def scoring_name(cell: str) -> str:
    return f"teacher-cell-{cell}-s0-ood01"


def cell_config(parent: dict, cell: str, gpu_uuid: str, selection_sha: str) -> dict:
    spec = CELLS[cell]
    config = copy.deepcopy(parent)
    config.update(
        campaign_id=campaign(cell),
        profile=spec["profile"],
        gpu_uuid=gpu_uuid,
        merged_model=f"/tmp/gleipnir-merged/{campaign(cell)}",
        training_python=TRAINING_PYTHON,
        extra_sources=sorted(
            str(p.relative_to(ROOT)) for p in HERE.iterdir() if p.suffix == ".py"
        ),
        **{k: v.format(cell=cell) for k, v in FIELDS.items()},
    )
    config["inputs"]["serving_selection"]["sha256"] = selection_sha
    config["inputs"]["startup_reference"] = {
        "path": spec["startup_reference"],
        "sha256": file_hash(ROOT / spec["startup_reference"]),
    }
    return config


def scoring_config(cell_cfg: dict, cell: str) -> dict:
    config = copy.deepcopy(cell_cfg)
    config.update(
        campaign_id=scoring_name(cell),
        merged_model=f"/tmp/gleipnir-merged/{scoring_name(cell)}",
        parent_campaign=f"results/{campaign(cell)}",
        score_splits=["ood"],
        evaluations=[],
    )
    config["evaluation"]["populations"] = {"ood": 6395}
    for key, (path, _) in OOD.items():
        config["inputs"][key] = {"path": path, "sha256": file_hash(ROOT / path)}
    return config


def prepare(cell: str, gpu_uuid: str) -> dict:
    if not gpu_uuid.startswith("GPU-"):
        raise ValueError("a verified live GPU UUID is required")
    parent = yaml.safe_load((ROOT / "data" / PARENT / "config.yaml").read_text())
    for name, spec in parent["inputs"].items():
        if name in ("serving_selection", "startup_reference"):
            continue
        if file_hash(ROOT / spec["path"]) != spec["sha256"]:
            raise ValueError(f"parent input drift: {name}")
    frozen = json.loads((ROOT / OOD_MANIFEST).read_text())["files_sha256"]
    for key, (path, expected) in OOD.items():
        actual = file_hash(ROOT / path)
        if (expected and actual != expected) or (
            key == "ood_workload" and frozen[path] != actual
        ):
            raise ValueError(f"OOD input drift: {key}")
    selection_sha = file_hash(ROOT / SELECTION)
    train = cell_config(parent, cell, gpu_uuid, selection_sha)
    score = scoring_config(train, cell)
    written = {}
    for name, config in ((campaign(cell), train), (scoring_name(cell), score)):
        data = ROOT / "data" / name
        if data.exists():
            raise ValueError(f"refuse to overwrite {data}")
        data.mkdir(parents=True)
        (data / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        written[name] = str((data / "config.yaml").relative_to(ROOT))
    path = ROOT / written[scoring_name(cell)]
    ctx = Campaign(ROOT, path, yaml.safe_load(path.read_text()))
    write_json(
        ROOT / "data" / scoring_name(cell) / "manifest.json",
        {
            "files_sha256": {
                written[scoring_name(cell)]: file_hash(path),
                **{s["path"]: s["sha256"] for s in ctx.config["inputs"].values()},
            },
            "source_sha256": ctx.sources(),
            "parent": score["parent_campaign"],
        },
    )
    report = {
        "status": "prepared",
        "cell": cell,
        "gpu_uuid": gpu_uuid,
        "configs": written,
    }
    write_json(
        ROOT / f"results/teacher_precision_factorial/preparation_{cell}.json", report
    )
    print("cell_prepared", json.dumps(written), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", choices=sorted(CELLS), required=True)
    parser.add_argument("--gpu-uuid", required=True)
    args = parser.parse_args()
    prepare(args.cell, args.gpu_uuid)
