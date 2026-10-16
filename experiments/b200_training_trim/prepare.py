"""Freeze the trimmed-final-layer training run against the seed-1 control."""

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
ARM = "training-trim-s1-01"
# Same GPU, same seed-1 partitions; arithmetic differs only in the loss scalar.
TIMING_CONTROLS = (
    "kd-temperature-t2-01",
    "kd-temperature-t4-01",
    "filtered-hard030-01",
)
FIELDS = {
    "hypothesis": (
        "Running the final full-attention layer only at packed decision positions "
        "keeps gradients within 1% and shortens steady updates by at least 2%."
    ),
    "intervention": "Seed-1 strict-filtered BF16/SDPA campaign with last_layer_trim.",
    "selection": (
        "Systems check only: trim canary, matched per-update timing and ID score "
        "agreement with teacher-seed-sdpa-s1-01; adoption is the user's decision."
    ),
    "stop_condition": (
        "Stop on a failed trim canary, input/runtime drift, nonfinite or missing "
        "gradients, failed master/merge/serving parity, incomplete scoring or "
        "completion."
    ),
}


def arm_config(control: dict, gpu_uuid: str, selection: str) -> dict:
    config = copy.deepcopy(control)
    config.update(
        campaign_id=ARM,
        gpu_uuid=gpu_uuid,
        merged_model=f"/tmp/gleipnir-merged/{ARM}",
        extra_sources=sorted(
            str(p.relative_to(ROOT)) for p in HERE.iterdir() if p.suffix == ".py"
        )
        + ["src/gleipnir/training/last_layer_trim.py"],
        last_layer_trim=True,
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
        "config": str((data / "config.yaml").relative_to(ROOT)),
    }
    write_json(ROOT / "results/b200_training_trim/preparation.json", report)
    print("training_trim_prepared", json.dumps(report), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid", required=True)
    prepare(parser.parse_args().gpu_uuid)
