"""Prevent training drift and false evaluation reuse in the historical replay."""

import json
from types import SimpleNamespace

import pytest
import yaml

from experiments.b200_augmented_sdpa_replay.run import (
    CONFIG,
    adapter_identity,
    historical_job_comparison,
)
from gleipnir.campaigns.monitoring import training
from gleipnir.campaigns.monitoring.contract import Campaign


def test_historical_job_rejects_arithmetic_change(tmp_path):
    old = {"job_name": "old", "learning_rate": 5e-5}
    path = tmp_path / "job.json"
    path.write_text(json.dumps(old))
    job = {**old, "job_name": "new", "expected_initial_master_sha256": "initial"}
    ctx = SimpleNamespace(
        input=lambda name: path,
        job=lambda: job,
        config={"model": {"initial_tensor_sha256": "initial"}},
    )
    assert historical_job_comparison(ctx)["passed"]
    job["learning_rate"] = 1e-4
    with pytest.raises(ValueError, match="recipe drift"):
        historical_job_comparison(ctx)


@pytest.mark.parametrize("change", [None, "weights", "config"])
def test_score_reuse_requires_both_weight_files_and_equivalent_configs(
    tmp_path, change
):
    adapter, original = tmp_path / "adapter", tmp_path / "original"
    inputs = {}
    for folder, prefix in (("causal_adapter", "master"), ("model", "export")):
        (adapter / folder).mkdir(parents=True)
        (original / folder).mkdir(parents=True)
        for root in (adapter, original):
            (root / folder / "adapter_model.safetensors").write_bytes(folder.encode())
            (root / folder / "adapter_config.json").write_text(
                json.dumps({"r": 128, "target_modules": ["q_proj", "v_proj"]})
            )
        inputs[f"historical_{prefix}"] = original / folder / "adapter_model.safetensors"
        inputs[f"historical_{prefix}_config"] = (
            original / folder / "adapter_config.json"
        )
    config = adapter / "model/adapter_config.json"
    config.write_text(json.dumps({"r": 128, "target_modules": ["v_proj", "q_proj"]}))
    if change == "weights":
        (adapter / "model/adapter_model.safetensors").write_bytes(b"different")
    if change == "config":
        config.write_text(json.dumps({"r": 64, "target_modules": ["v_proj", "q_proj"]}))
    ctx = SimpleNamespace(adapter=adapter, input=lambda name: inputs[name])
    assert adapter_identity(ctx)["identical"] is (change is None)


def test_legacy_sdpa_profile_binds_reference_and_routes_startup_defaults(
    tmp_path, monkeypatch
):
    config = yaml.safe_load(CONFIG.read_text())
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    ctx = Campaign.load(tmp_path, path)
    job = ctx.job()
    assert (
        job["startup_validation_reference_sha256"]
        == config["inputs"]["startup_reference"]["sha256"]
    )
    assert "packed_attention_backend" not in job
    monkeypatch.setattr(Campaign, "check", lambda self: None)

    class StartupReached(Exception):
        pass

    def reference(path, **kwargs):
        assert kwargs["packed_attention_backend"] == "sdpa"
        assert kwargs["packed_attention_version"] is None
        assert kwargs["learning_gradient_tolerance"] is None
        raise StartupReached

    monkeypatch.setattr("gleipnir.training.startup.validation_reference", reference)
    with pytest.raises(StartupReached):
        training.train(ctx)
