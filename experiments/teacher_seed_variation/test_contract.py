"""The seed-1 campaign may differ from seed 0 only in declared fields."""

from pathlib import Path

import yaml

from experiments.teacher_seed_variation.prepare import (
    FIELDS,
    PARENT,
    ROOT,
    SCORING,
    scoring_config,
    scoring_plan,
    seed_config,
)

ALLOWED = {
    "campaign_id",
    "seed",
    "gpu_uuid",
    "merged_model",
    "extra_sources",
    "training_python",
    *FIELDS,
}


def parent() -> dict:
    return yaml.safe_load((ROOT / "data" / PARENT / "config.yaml").read_text())


def test_only_seed_identity_and_serving_hash_change():
    base = parent()
    config = seed_config(base, "GPU-test", "0" * 64)
    changed = {k for k in config if config[k] != base.get(k)} - {"inputs"}
    assert changed <= ALLOWED and config["seed"] == 1 and base["seed"] == 0
    for name, spec in config["inputs"].items():
        if name != "serving_selection":
            assert spec == base["inputs"][name]
    for key in ("profile", "learning_rate", "model", "logical_batch_size"):
        assert config[key] == base[key]
    assert (
        config["model"]["initial_tensor_sha256"]
        == base["model"]["initial_tensor_sha256"]
    )


def test_scoring_contexts_bind_fixed_parents_and_splits():
    seed = seed_config(parent(), "GPU-test", "0" * 64)
    for name, (path, splits) in SCORING.items():
        config = scoring_config(seed, name)
        assert config["parent_campaign"] == path
        assert config["score_splits"] == splits
        assert set(config["evaluation"]["populations"]) == set(splits)
        assert Path(config["merged_model"]).name == name
        assert ("id" in config["evaluations"]) == ("id" in splits)


def test_later_seeds_change_only_seed_identity_and_score_ood():
    base = parent()
    config = seed_config(base, "GPU-test", "0" * 64, seed=2)
    assert config["seed"] == 2 and config["campaign_id"] == "teacher-seed-sdpa-s2-01"
    assert "seed 2" in config["intervention"]
    plan = scoring_plan(2)
    assert list(plan) == ["teacher-seed-sdpa-s2-ood01"]
    scoring = scoring_config(config, "teacher-seed-sdpa-s2-ood01")
    assert scoring["evaluations"] == [] and scoring["score_splits"] == ["ood"]
    assert scoring["parent_campaign"] == "results/teacher-seed-sdpa-s2-01"
