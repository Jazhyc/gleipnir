"""Factorial cells change only the training profile and startup reference."""

import yaml

from experiments.teacher_precision_factorial.prepare import (
    CELLS,
    FIELDS,
    ROOT,
    cell_config,
    scoring_config,
)
from experiments.teacher_seed_variation.prepare import PARENT

ALLOWED = {
    "campaign_id",
    "profile",
    "gpu_uuid",
    "merged_model",
    "training_python",
    "extra_sources",
    *FIELDS,
}


def test_cells_change_only_profile_and_startup_reference():
    parent = yaml.safe_load((ROOT / "data" / PARENT / "config.yaml").read_text())
    for cell, spec in CELLS.items():
        config = cell_config(parent, cell, "GPU-test", "0" * 64)
        changed = {k for k in config if config[k] != parent.get(k)} - {"inputs"}
        assert changed <= ALLOWED
        assert config["profile"] == spec["profile"]
        assert config["seed"] == parent["seed"] == 0
        for name, value in config["inputs"].items():
            if name not in ("serving_selection", "startup_reference"):
                assert value == parent["inputs"][name]
        score = scoring_config(config, cell)
        assert score["evaluations"] == [] and score["score_splits"] == ["ood"]
