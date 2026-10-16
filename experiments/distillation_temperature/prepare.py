"""Freeze the one-seed distillation-temperature arms against the seed-1 control."""

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
# Frozen before training (README). Each arm changes exactly one loss knob.
ARMS = {
    "kd-temperature-t2-01": {"soft_kd_temperature": 2.0},
    "kd-temperature-t4-01": {"soft_kd_temperature": 4.0},
    "kd-sharpen-t050-01": {"soft_target_logit_scale": 0.5},
}
FIELDS = {
    "hypothesis": (
        "Changing the distillation temperature of the Kimi K3 binary soft loss "
        "improves held-out ID macro pAUROC@20 over the seed-matched T=1 control."
    ),
    "intervention": "Seed-1 strict-filtered BF16/SDPA campaign with only {knob}.",
    "selection": (
        "Final one-epoch checkpoint; ID-only paired comparison with "
        "teacher-seed-sdpa-s1-01; advance to more seeds, never promote, on one seed."
    ),
    "stop_condition": (
        "Stop on input/runtime drift, nonfinite or missing gradients, failed "
        "master/merge/serving parity, incomplete scoring or completion."
    ),
}


def arm_config(control: dict, name: str, knob: dict, gpu_uuid: str, selection: str):
    """Change only identity, GPU, merge slot, serving hash, contract text and knob."""
    if set(knob) - {"soft_kd_temperature", "soft_target_logit_scale"} or len(knob) != 1:
        raise ValueError("each arm changes exactly one temperature knob")
    config = copy.deepcopy(control)
    text = ", ".join(f"{k}={v}" for k, v in knob.items())
    config.update(
        campaign_id=name,
        gpu_uuid=gpu_uuid,
        merged_model=f"/tmp/gleipnir-merged/{name}",
        extra_sources=sorted(
            str(p.relative_to(ROOT)) for p in HERE.iterdir() if p.suffix == ".py"
        ),
        **{k: v.format(knob=text) for k, v in FIELDS.items()},
        **knob,
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
    selection = file_hash(ROOT / SELECTION)
    written = {}
    for name, knob in ARMS.items():
        data = ROOT / "data" / name
        if data.exists():
            raise ValueError(f"refuse to overwrite {data}")
        data.mkdir(parents=True)
        config = arm_config(control, name, knob, gpu_uuid, selection)
        (data / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        written[name] = str((data / "config.yaml").relative_to(ROOT))
    report = {
        "status": "prepared",
        "gpu_uuid": gpu_uuid,
        "control_config_sha256": file_hash(ROOT / "data" / CONTROL / "config.yaml"),
        "arms": ARMS,
        "configs": written,
    }
    write_json(ROOT / "results/distillation_temperature/preparation.json", report)
    print("distillation_temperature_prepared", json.dumps(written), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid", required=True)
    prepare(parser.parse_args().gpu_uuid)
