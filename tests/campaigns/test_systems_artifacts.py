"""Latest-only diagnostic weights retain immutable receipts and survive failures."""

import pytest
import torch

from gleipnir.systems_artifacts import save_systems_master, systems_scratch


def test_scratch_output_is_explicit_and_does_not_change_historical_jobs():
    from pathlib import Path

    from gleipnir.monitoring_systems_screen import (
        load_config,
        make_jobs_unchecked,
        resolve_paths,
    )
    from gleipnir.monitoring_training_command import training_command

    config = load_config(Path("experiments/b200_training_throughput/config.yaml"))
    paths = resolve_paths(config)
    original = make_jobs_unchecked(config, paths, "selection_hash")
    assert all("systems_adapter_scratch" not in job for job in original)
    config["systems_adapter_scratch"] = True
    scratch = make_jobs_unchecked(config, paths, "selection_hash")
    for old, new in zip(original, scratch, strict=True):
        assert {k: v for k, v in new.items() if k != "systems_adapter_scratch"} == old
        assert "++student.training.systems_adapter_scratch=true" in training_command(
            new
        )


def test_new_trial_replaces_same_master_without_accumulating_weights(tmp_path):
    first = save_systems_master({"lora": torch.tensor([1.0])}, tmp_path)
    second = save_systems_master({"lora": torch.tensor([2.0])}, tmp_path)
    assert first == second
    assert second["mutable"] is True
    directory = systems_scratch(tmp_path)
    assert list(directory.iterdir()) == [directory / "fp32_master.pt"]
    assert (
        torch.load(directory / "fp32_master.pt", weights_only=True)["lora"].item()
        == 2.0
    )


def test_failed_save_preserves_previous_master(tmp_path, monkeypatch):
    save_systems_master({"lora": torch.tensor([1.0])}, tmp_path)

    def failing_save(value, path):
        path.write_bytes(b"incomplete")
        raise OSError("simulated storage error")

    monkeypatch.setattr(torch, "save", failing_save)
    with pytest.raises(OSError, match="storage error"):
        save_systems_master({"lora": torch.tensor([2.0])}, tmp_path)
    directory = systems_scratch(tmp_path)
    assert list(directory.iterdir()) == [directory / "fp32_master.pt"]
    assert (
        torch.load(directory / "fp32_master.pt", weights_only=True)["lora"].item()
        == 1.0
    )
