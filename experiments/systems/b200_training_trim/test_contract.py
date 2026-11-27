from pathlib import Path

import yaml

from experiments.systems.b200_training_trim.prepare import CONTROL, arm_config

ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()
)


def test_arm_adds_only_the_trim_to_the_control() -> None:
    control = yaml.safe_load((ROOT / "data" / CONTROL / "config.yaml").read_text())
    arm = arm_config(control, "GPU-test", "selection-sha")
    changed = {k for k in set(arm) | set(control) if arm.get(k) != control.get(k)}
    assert changed == {
        "campaign_id",
        "extra_sources",
        "gpu_uuid",
        "merged_model",
        "inputs",
        "hypothesis",
        "intervention",
        "selection",
        "stop_condition",
        "last_layer_trim",
    }
    assert arm["last_layer_trim"] is True
