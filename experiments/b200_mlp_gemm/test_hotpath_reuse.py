"""Validation reuse is bound to the same completed worker and input contract."""

import json

import pytest

from experiments.b200_mlp_gemm.hotpath_reuse import reusable_validation


@pytest.mark.parametrize(
    "change",
    [
        None, "source", "installer", "normalize", "async", "metadata",
        "pid", "warm", "physical",
    ],
)
def test_reuse_requires_matching_completed_integration(tmp_path, change):
    trial = tmp_path / "13profile"
    trial.mkdir()
    source = "PROFILE = True\ndef installed(trainer):\n    return trainer\n"
    validation = {
        "accepted_for_timing": True,
        "integration_source_sha256": "source",
        "installation": {
            "normalization_input_copy_removed": False,
            "nonblocking_trainer_inputs": True,
            "cpu_prepared_packing": True,
        },
        "initial_master_sha256": "master",
        "masters_unchanged": True,
        "optimizer_updates": 0,
    }
    receipt = {
        "status": "complete",
        "variant": "candidate",
        "pid": 123,
        "initial_master_sha256": "master",
        "physical_contract": [{"tokens": 5}],
        "measured_updates_warm": True,
        "step_seconds": [1.0] * 20,
    }
    if change == "source":
        validation["integration_source_sha256"] = "other"
    if change == "installer":
        source = source.replace("return trainer", "return None")
    if change == "normalize":
        validation["installation"]["normalization_input_copy_removed"] = True
    if change == "async":
        validation["installation"]["nonblocking_trainer_inputs"] = False
    if change == "metadata":
        validation["installation"]["cpu_prepared_packing"] = False
    if change == "pid":
        receipt["pid"] = 124
    if change == "warm":
        receipt["measured_updates_warm"] = False
    if change == "physical":
        receipt["physical_contract"] = [{"tokens": 6}]
    (trial / "candidate_validation.json").write_text(json.dumps(validation))
    (trial / "receipt.json").write_text(json.dumps(receipt))
    (trial / "executed_hotpath_candidate.py").write_text(source)
    found = reusable_validation(
        tmp_path,
        integration_sha256="source",
        installer_source=(
            "PROFILE = False\ndef installed(trainer):\n    return trainer\n"
        ),
        normalize=False,
        async_inputs=True,
        prepared_metadata=True,
        worker_pid=123,
        initial_master="master",
        physical_contract=[{"tokens": 5}],
    )
    assert (found is not None) is (change is None)
    if found:
        assert not found["performed_this_trial"]
        assert len(found["reuse_reference_sha256"]) == 64
