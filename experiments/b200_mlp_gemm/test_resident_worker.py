"""Resident trials must reset state without destroying compiled parameter identity."""

from types import SimpleNamespace

import pytest
import torch

from experiments.b200_mlp_gemm.resident_worker import (
    capture_rng,
    reset_trainer,
    restore_rng,
    validate_request,
    write_json,
)
from experiments.b200_mlp_gemm.test_timing_screen import student_config
from gleipnir.packed_training import validate_packed_training_config
from gleipnir.training_execution_audit import tensor_digest


@pytest.mark.parametrize(
    "payload",
    [
        {"id": "../escape"},
        {"id": "trial", "variant": "exec"},
        {"id": 5},
    ],
)
def test_queue_rejects_unknown_conditions(payload):
    with pytest.raises(ValueError):
        validate_request(payload)


def test_queue_atomic_publish(tmp_path):
    import json

    path = tmp_path / "trial.json"
    validate_request({"id": "01baseline"})
    write_json(path, {"id": "01baseline"})
    assert json.loads(path.read_text()) == {"id": "01baseline"}
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("recover", [False, True])
def test_reset_preserves_parameters_and_restores_rng(monkeypatch, recover):
    from experiments.b200_mlp_gemm.resident_worker import recover_candidate

    model = torch.nn.Linear(4, 2)
    parameters = list(model.parameters())
    initial = [p.detach().clone() for p in parameters]
    timer = type("OptimizerStepTimer", (), {})()
    timer.durations, timer.started_at = [1.2], 3.0
    handler = SimpleNamespace(
        callbacks=[timer], optimizer=object(), lr_scheduler=object()
    )
    trainer = SimpleNamespace(
        model=model,
        optimizer=object(),
        lr_scheduler=object(),
        callback_handler=handler,
        _created_lr_scheduler=True,
    )
    rng = capture_rng()
    expected = torch.rand(4)
    with torch.no_grad():
        for p in parameters:
            p.add_(1)
    if recover:
        monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
        digest = recover_candidate(trainer, initial, rng, tensor_digest(initial))
    else:
        digest = reset_trainer(trainer, initial, rng)
    assert digest == tensor_digest(initial)
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
    assert torch.equal(torch.rand(4), expected)
    assert trainer.optimizer is trainer.lr_scheduler is None
    assert handler.optimizer is handler.lr_scheduler is None
    assert not timer.durations and timer.started_at is None


def test_recovery_rejects_unusable_cuda_context(monkeypatch):
    from experiments.b200_mlp_gemm.resident_worker import recover_candidate

    def failed_cuda():
        raise RuntimeError("CUDA context unusable")

    monkeypatch.setattr(torch.cuda, "synchronize", failed_cuda)
    with pytest.raises(RuntimeError, match="CUDA context unusable"):
        recover_candidate(None, [], {}, "unused")


def test_resident_validation_reuse_is_scoped():
    cfg = student_config()
    cfg["training"].update(
        startup_validation_reference="bound.json",
        startup_validation_reference_sha256="digest",
        packing_learning_gradient_tolerance=0.05,
        native_fp4_mlp_resident=True,
    )
    assert validate_packed_training_config(cfg)
    cfg["training"]["native_fp4_mlp_resident"] = "true"
    with pytest.raises(ValueError):
        validate_packed_training_config(cfg)
    cfg["training"]["native_fp4_mlp_resident"] = True
    cfg["training"]["max_steps"] = 21
    with pytest.raises(ValueError):
        validate_packed_training_config(cfg)


def test_rng_restore_is_repeatable():
    import random

    import numpy as np

    rng = capture_rng()
    first = (random.random(), np.random.rand(), torch.rand(1))
    restore_rng(rng)
    second = (random.random(), np.random.rand(), torch.rand(1))
    assert first[:2] == second[:2]
    assert torch.equal(first[2], second[2])


@pytest.mark.parametrize(
    "flag", ["GLEIPNIR_FP4_PROFILE_OUTPUT", "GLEIPNIR_FP4_RESIDENT_ROOT"]
)
def test_entry_scopes_validation_reuse_for_both_paths(monkeypatch, flag, tmp_path):
    from experiments.b200_mlp_gemm import fp4_training_entry
    from experiments.b200_mlp_gemm.full_model_profile import (
        profile_validation_reference,
    )
    from gleipnir import validated_startup

    original = validated_startup.validation_reference
    monkeypatch.delenv("GLEIPNIR_FP4_WARM_REPORT", raising=False)
    monkeypatch.delenv("GLEIPNIR_FP4_RUNTIME_REPORT", raising=False)
    monkeypatch.setenv(flag, str(tmp_path))

    def execute(*args, **kwargs):
        assert validated_startup.validation_reference is profile_validation_reference

    monkeypatch.setattr(fp4_training_entry.runpy, "run_path", execute)
    fp4_training_entry.main()
    assert validated_startup.validation_reference is original


def test_candidate_context_checks_hash_and_restores_intervention(monkeypatch, tmp_path):
    import hashlib

    from experiments.b200_mlp_gemm import resident_worker

    monkeypatch.setattr(resident_worker, "__file__", str(tmp_path / "worker.py"))
    source = b"""from contextlib import contextmanager
@contextmanager
def intervention(trainer):
    trainer.active = True
    try: yield
    finally: trainer.active = False
def validate(trainer):
    return {"accepted_for_timing": True, "arithmetic_changed": False}
"""
    (tmp_path / "resident_candidate.py").write_bytes(source)
    trial = tmp_path / "trial"
    trial.mkdir()
    trainer = SimpleNamespace(active=False)
    request = {"source_sha256": hashlib.sha256(source).hexdigest()}
    with pytest.raises(RuntimeError):
        with resident_worker.candidate_context(trainer, request, trial):
            assert trainer.active
            raise RuntimeError("failed trial")
    assert not trainer.active
    assert (trial / "candidate_source.py").read_bytes() == source
    request["source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum"):
        with resident_worker.candidate_context(trainer, request, trial):
            pytest.fail("must not execute changed candidate")


def test_control_import_does_not_load_training_stack():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "import experiments.b200_mlp_gemm.resident_launch; "
            "assert 'torch' not in sys.modules; "
            "assert 'transformers' not in sys.modules",
        ],
        check=True,
    )


def test_candidate_override_is_owned_and_records_execution(monkeypatch, tmp_path):
    import hashlib
    import json

    from experiments.b200_mlp_gemm import resident_worker

    worker = tmp_path / "experiments/worker/resident_worker.py"
    worker.parent.mkdir(parents=True)
    monkeypatch.setattr(resident_worker, "__file__", str(worker))
    candidate = tmp_path / "experiments/frost/candidate.py"
    candidate.parent.mkdir()
    candidate.write_text("""from contextlib import contextmanager
@contextmanager
def intervention(trainer):
    trainer.active = True
    try: yield
    finally: trainer.active = False
def validate(trainer):
    return {"accepted_for_timing": True}
def finish():
    return {"direct_calls": 17}
""")
    monkeypatch.setenv("GLEIPNIR_FP4_RESIDENT_CANDIDATE", str(candidate))
    trial = tmp_path / "trial"
    trial.mkdir()
    request = {"source_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest()}
    trainer = SimpleNamespace(active=False)
    with resident_worker.candidate_context(trainer, request, trial):
        assert trainer.active
    assert not trainer.active
    assert json.loads((trial / "candidate_execution.json").read_text()) == {
        "direct_calls": 17
    }
    monkeypatch.setenv("GLEIPNIR_FP4_RESIDENT_CANDIDATE", str(tmp_path / "outside.py"))
    with pytest.raises(ValueError, match="experiment-owned"):
        with resident_worker.candidate_context(trainer, request, trial):
            pytest.fail("must not load outside source")
