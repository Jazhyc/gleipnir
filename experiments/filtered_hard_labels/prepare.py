"""Freeze the one-seed 0.3 hard-label arm on the strict-filtered recipe."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from gleipnir.data.monitoring import file_hash, write_json

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CONTROL = "teacher-seed-sdpa-s1-01"
SELECTION = "experiments/b200_inference_benchmark/serving_default.json"
ARM = "filtered-hard030-01"
# Old screen's normalized mixture: soft (1 - f), source-label CE f.
HARD_FRACTION = 0.3
FIELDS = {
    "hypothesis": (
        "A 0.3 source-label hard-loss fraction still improves held-out ID macro "
        "pAUROC@20 on the strict-filtered recipe over the seed-matched soft-only "
        "control."
    ),
    "intervention": (
        "Seed-1 strict-filtered BF16/SDPA campaign with soft_loss_weight 0.7 and "
        "direct_loss_weight 0.3 only; LR, data, targets and schedule unchanged."
    ),
    "selection": (
        "Final one-epoch checkpoint; ID-only paired comparison with "
        "teacher-seed-sdpa-s1-01; advance to more seeds, never promote, on one seed."
    ),
    "stop_condition": (
        "Stop on input/runtime drift, nonfinite or missing gradients, failed "
        "master/merge/serving parity, incomplete scoring or completion."
    ),
}


def arm_config(control: dict, gpu_uuid: str, selection: str) -> dict:
    """Change only identity, GPU, merge slot, serving hash, text and loss weights."""
    if control["soft_loss_weight"] != 1.0 or control["direct_loss_weight"] != 0.0:
        raise ValueError("control is not the soft-only recipe")
    config = copy.deepcopy(control)
    config.update(
        campaign_id=ARM,
        gpu_uuid=gpu_uuid,
        merged_model=f"/tmp/gleipnir-merged/{ARM}",
        extra_sources=sorted(
            str(p.relative_to(ROOT)) for p in HERE.iterdir() if p.suffix == ".py"
        ),
        soft_loss_weight=1.0 - HARD_FRACTION,
        direct_loss_weight=HARD_FRACTION,
        **FIELDS,
    )
    config["inputs"]["serving_selection"]["sha256"] = selection
    return config


def prepare(gpu_uuid: str) -> dict:
    if not gpu_uuid.startswith("GPU-"):
        raise ValueError("a verified live GPU UUID is required")
    control = yaml.safe_load((ROOT / "data" / CONTROL / "config.yaml").read_text())
    for name, spec in control["inputs"].items():
        if (
            name != "serving_selection"
            and file_hash(ROOT / spec["path"]) != spec["sha256"]
        ):
            raise ValueError(f"control input drift: {name}")
    data = ROOT / "data" / ARM
    if data.exists():
        raise ValueError(f"refuse to overwrite {data}")
    data.mkdir(parents=True)
    config = arm_config(control, gpu_uuid, file_hash(ROOT / SELECTION))
    (data / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    report = {
        "status": "prepared",
        "gpu_uuid": gpu_uuid,
        "control_config_sha256": file_hash(ROOT / "data" / CONTROL / "config.yaml"),
        "hard_fraction": HARD_FRACTION,
        "config": str((data / "config.yaml").relative_to(ROOT)),
    }
    write_json(ROOT / "results/filtered_hard_labels/preparation.json", report)
    print("filtered_hard_labels_prepared", json.dumps(report), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid", required=True)
    prepare(parser.parse_args().gpu_uuid)
