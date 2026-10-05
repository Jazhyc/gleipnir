"""Finite timing acceptance must bind the changed precision/source/worker."""

import hashlib
import json
import os
from pathlib import Path

import pytest

from experiments.b200_mlp_gemm import grouped_candidate as candidate


@pytest.fixture
def previous(tmp_path, monkeypatch):
    monkeypatch.setattr(candidate, "FINITE_TIMING_REFERENCE", "previous")
    previous = tmp_path / "previous"
    previous.mkdir()
    (previous / "executed_grouped_candidate.py").write_text(
        Path(candidate.__file__).read_text()
    )
    receipt = {
        "worker_pid": os.getpid(),
        "initial_master_sha256": "master",
        "masters_unchanged": True,
        "optimizer_updates": 0,
        "physical_contract_agreement": True,
        "integration_source_sha256": hashlib.sha256(
            Path(candidate.integration.__file__).read_bytes()
        ).hexdigest(),
        "baseline_first_batch_loss": 0.52,
        "candidate_first_batch_loss": 0.52,
        "adapter_gradient_relative_l2": 0.18,
        "accepted_for_timing": False,
    }
    (previous / "grouped_validation.json").write_text(json.dumps(receipt))
    (previous / "failure.json").write_text(json.dumps({"baseline_restored": True}))
    return previous, receipt


def test_failed_strict_result_is_preserved_as_timing_only(previous):
    path, _ = previous
    actual = candidate.finite_timing_validation(path.parent / "new", "master")
    assert actual["accepted_for_timing"] is True
    assert actual["strict_parity_passed"] is False
    assert actual["accepted_for_training_replacement"] is False
    assert actual["performed_this_trial"] is False
    assert actual["adapter_gradient_relative_l2"] == 0.18
    assert len(actual["reuse_reference_sha256"]) == 64
    assert (
        json.loads((path / "grouped_validation.json").read_text())[
            "accepted_for_timing"
        ]
        is False
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("worker_pid", -1),
        ("initial_master_sha256", "changed"),
        ("integration_source_sha256", "changed"),
        ("masters_unchanged", False),
        ("physical_contract_agreement", False),
        ("optimizer_updates", 1),
    ],
)
def test_identity_drift_rejected(previous, key, value):
    path, receipt = previous
    receipt[key] = value
    (path / "grouped_validation.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="identity mismatch"):
        candidate.finite_timing_validation(path.parent / "new", "master")


def test_changed_installer_rejected(previous):
    path, _ = previous
    source = path / "executed_grouped_candidate.py"
    source.write_text(source.read_text().replace("yield\n", "yield 123\n"))
    with pytest.raises(ValueError, match="identity mismatch"):
        candidate.finite_timing_validation(path.parent / "new", "master")
