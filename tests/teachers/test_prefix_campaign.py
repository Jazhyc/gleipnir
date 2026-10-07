import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from experiments.monitoring_prefix_supervision import campaign
from experiments.monitoring_prefix_supervision.campaign import make_jobs


def test_prefix_jobs_preserve_parent_budget_and_full_targets():
    config = OmegaConf.to_container(
        OmegaConf.load(Path("experiments/monitoring_prefix_supervision/training.yaml")),
        resolve=True,
    )
    paired = {
        "parent_rows": 8688,
        "seed": 0,
        "epoch": 0,
        "output_sha256": "paired",
        "cache_contract_sha256": "teacher",
        "parents_with_prefix": 6348,
    }
    jobs = make_jobs(config, paired)
    assert [j["prefix_loss_weight"] for j in jobs] == [0.25, 0.5]
    for job in jobs:
        assert job["train_rows"] == 8688
        assert job["learning_rate"] == 2e-5
        assert job["num_train_epochs"] == 1
        assert job["micro_batch_size"] * job["gradient_accumulation_steps"] == 32
        assert job["soft_targets"] == config["full_soft_targets"]
        assert job["student_rows_sha256"] == "paired"
        assert job["save_steps"] == 272
    with pytest.raises(ValueError, match="design"):
        make_jobs({**config, "epochs": 2}, paired)
    low = make_jobs({**config, "prefix_loss_weights": [0.05, 0.1]}, paired)
    assert [j["job_name"] for j in low] == [
        "prefix-w005-lr2em05-seed0",
        "prefix-w010-lr2em05-seed0",
    ]
    for original, followup in zip(jobs, low, strict=True):
        for key in ("student_rows_sha256", "soft_targets", "save_steps", "seed"):
            assert followup[key] == original[key]
    with pytest.raises(ValueError, match="design"):
        make_jobs({**config, "prefix_loss_weights": [0.05, 0.2]}, paired)


def test_execution_rejects_inference_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(campaign.importlib.metadata, "version", lambda _: "2.13.0")
    with pytest.raises(ValueError, match="preserved training"):
        campaign.execute(tmp_path, None)


def test_execution_does_not_interrupt_busy_gpus(monkeypatch, tmp_path):
    monkeypatch.setattr(
        campaign.importlib.metadata, "version", lambda _: "2.11.0+cu130"
    )
    monkeypatch.setattr(
        campaign.subprocess, "check_output", lambda *a, **k: "75583\n75583\n"
    )
    with pytest.raises(RuntimeError, match="do not interrupt"):
        campaign.execute(tmp_path, None)


def test_reuse_requires_exact_previous_provenance(tmp_path):
    rows = tmp_path / "paired.jsonl"
    rows.write_text('{"id": "a"}\n')
    sidecar = rows.with_suffix(".jsonl.manifest.json")
    paired = {
        "output_sha256": campaign.sha256_file(rows),
        "numerical_exception": {"a": 1},
    }
    sidecar.write_text(json.dumps(paired))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "files": {str(p): campaign.sha256_file(p) for p in (rows, sidecar)},
                "paired_training": paired,
            }
        )
    )
    config = {
        "paired_rows": str(rows),
        "reuse_campaign_manifest": str(manifest),
        "numerical_exception": {"a": 1},
    }
    assert campaign.reuse_paired(config) == paired
    with pytest.raises(ValueError, match="provenance"):
        campaign.reuse_paired({**config, "numerical_exception": None})
    rows.write_text("changed")
    with pytest.raises(ValueError, match="drift"):
        campaign.reuse_paired(config)
