"""Candidate validation reuse requires matching source and a warm completed run."""

import json

import pytest

from experiments.b200_mlp_gemm.candidate_reuse import reusable_validation


@pytest.mark.parametrize(
    "change",
    [None, "source", "variant", "master", "physical", "warm", "complete", "pid"],
)
def test_validation_identity_and_success_are_required(tmp_path, change):
    trial = tmp_path / "04candidate"
    trial.mkdir()
    validation = {
        "accepted_for_timing": True,
        "integration_source_sha256": "a" * 64,
        "installation": {"merged_qkv_z": True},
        "initial_master_sha256": "master",
        "masters_unchanged": True,
        "optimizer_updates": 0,
    }
    result = {
        "status": "complete",
        "variant": "candidate",
        "pid": 123,
        "initial_master_sha256": "master",
        "physical_contract": [{"tokens": 10}],
        "measured_updates_warm": True,
        "step_seconds": [1.0] * 20,
    }
    if change == "source":
        validation["integration_source_sha256"] = "b" * 64
    elif change == "variant":
        validation["installation"]["merged_qkv_z"] = False
    elif change == "master":
        validation["initial_master_sha256"] = "other"
    elif change == "physical":
        result["physical_contract"] = [{"tokens": 11}]
    elif change == "warm":
        result["measured_updates_warm"] = False
    elif change == "complete":
        result["status"] = "failed"
    elif change == "pid":
        result["pid"] = 124
    (trial / "candidate_validation.json").write_text(json.dumps(validation))
    (trial / "receipt.json").write_text(json.dumps(result))
    found = reusable_validation(
        tmp_path,
        integration_sha256="a" * 64,
        merged_inputs=True,
        worker_pid=123,
        initial_master="master",
        physical_contract=[{"tokens": 10}],
    )
    assert (found is not None) is (change is None)
    if found:
        assert found["validation"] == validation
        assert len(found["sha256"]) == 64
