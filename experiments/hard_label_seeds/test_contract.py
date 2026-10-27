"""Hard seeds may differ from the soft seed-0 parent only in declared fields."""

from pathlib import Path

import yaml

from experiments.hard_label_seeds.prepare import (
    FIELDS,
    PARENT,
    ROOT,
    campaign,
    ood_context,
    scoring_config,
    seed_config,
)

ALLOWED = {
    "campaign_id",
    "seed",
    "soft_loss_weight",
    "direct_loss_weight",
    "gpu_uuid",
    "merged_model",
    "extra_sources",
    "training_python",
    *FIELDS,
}


def parent() -> dict:
    return yaml.safe_load((ROOT / "data" / PARENT / "config.yaml").read_text())


def test_only_seed_objective_identity_and_serving_hash_change():
    base = parent()
    assert (base["soft_loss_weight"], base["direct_loss_weight"]) == (1.0, 0.0)
    for seed in (0, 1, 2):
        config = seed_config(base, "GPU-test", "0" * 64, seed)
        changed = {k for k in config if config[k] != base.get(k)} - {"inputs"}
        assert changed <= ALLOWED
        assert config["seed"] == seed and config["campaign_id"] == campaign(seed)
        assert (config["soft_loss_weight"], config["direct_loss_weight"]) == (0.0, 1.0)
        for name, spec in config["inputs"].items():
            if name != "serving_selection":
                assert spec == base["inputs"][name]
        for key in ("profile", "learning_rate", "model", "logical_batch_size"):
            assert config[key] == base[key]


def test_ood_contexts_bind_their_hard_parent():
    for seed in (0, 1, 2):
        config = scoring_config(seed_config(parent(), "GPU-test", "0" * 64, seed))
        assert config["parent_campaign"] == f"results/{campaign(seed)}"
        assert config["score_splits"] == ["ood"] and config["evaluations"] == []
        assert Path(config["merged_model"]).name == ood_context(seed)
        assert config["soft_loss_weight"] == 0.0
