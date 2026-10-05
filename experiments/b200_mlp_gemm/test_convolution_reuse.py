"""Only completed, identical arithmetic preparation can survive an audit fix."""

import pytest

from experiments.b200_mlp_gemm.convolution_reuse import can_reuse_preparation


@pytest.mark.parametrize(
    "change",
    [None, "kernel", "integration", "master", "steps", "restored", "pid", "updates"],
)
def test_preparation_reuse_requires_identical_arithmetic_and_reset(change):
    sources = {
        "candidate": "new driver",
        "reuse": "new helper",
        "integration": "adapter",
        "native.py": "kernel",
    }
    validation = {
        "accepted_for_timing": True,
        "quality_selection_eligible": False,
        "initial_master_sha256": "master",
        "masters_unchanged": True,
        "optimizer_updates": 0,
        "steps": [{}] * 20,
        "source_sha256": {
            "candidate": "old driver",
            "integration": "adapter",
            "native.py": "kernel",
        },
    }
    failure = {"baseline_restored": True, "initial_master_sha256": "master"}
    worker_pid = 123
    if change == "kernel":
        sources["native.py"] = "other"
    if change == "integration":
        sources["integration"] = "other"
    if change == "master":
        validation["initial_master_sha256"] = "other"
    if change == "steps":
        validation["steps"] = [{}] * 19
    if change == "restored":
        failure["baseline_restored"] = False
    if change == "pid":
        worker_pid = 124
    if change == "updates":
        validation["optimizer_updates"] = 1
    assert can_reuse_preparation(
        validation,
        failure,
        sources,
        initial_master="master",
        worker_pid=worker_pid,
        baseline_pid=123,
    ) is (change is None)
