"""Freeze source transformation boundaries and runtime scope restoration."""

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.native_fp8_qat import runtime
from experiments.native_fp8_qat.generate import transform
from experiments.native_fp8_qat.run import stage_environment


@pytest.mark.parametrize("stage", ["train", "reference", "evaluate"])
def test_frozen_template_derivative_changes_only_declared_substitutions(stage):
    source = Path(f"experiments/quantization_recovery/{stage}.py").read_text()
    generated, changes = transform(source, stage)
    restored = generated
    for change in reversed(changes):
        assert change["count"] == 1
        restored = restored.replace(change["new"], change["old"])
    assert restored == source
    assert 'config["teacher_experiment"]' in generated
    assert "experiments/native_fp8_qat/config.yaml" in generated
    if stage == "train":
        assert "verify_operator_reference(root, output)" in generated
        assert "decoder_canary()" not in generated
        assert 'config["experiment"]' in generated


def test_template_drift_fails_closed():
    with pytest.raises(ValueError, match="substitution count"):
        transform("print('missing original contract')", "train")


def test_archived_stage_environment_preserves_pinned_overlays_and_project_imports():
    original = {"PYTHONPATH": "/pinned/overlay:/project/src", "OTHER": "preserved"}
    actual = stage_environment(Path("/project"), original)
    assert actual["PYTHONPATH"] == "/project:/pinned/overlay:/project/src"
    assert actual["OTHER"] == "preserved"
    assert original["PYTHONPATH"].startswith("/pinned/overlay")


def test_variant_context_enables_native_fp8_and_restores_after_failure(monkeypatch):
    seen = []

    @contextmanager
    def scope(*args, **kwargs):
        seen.append((args, kwargs))
        try:
            yield
        finally:
            seen.append("restored")

    model = SimpleNamespace()
    monkeypatch.setattr(runtime, "_MODEL", model)
    monkeypatch.setattr(runtime.reference, "attention_runtime", scope)
    monkeypatch.setattr(runtime, "optimized_path", scope)
    with pytest.raises(RuntimeError), runtime.attention_runtime():
        raise RuntimeError("diagnostic failure")
    assert seen[1] == ((model,), {"fp8": True, "mlp": True, "gdn": True})
    assert seen[-2:] == ["restored", "restored"]


def test_runtime_requires_installed_model(monkeypatch):
    monkeypatch.setattr(runtime, "_MODEL", None)
    with pytest.raises(RuntimeError, match="install"):
        with runtime.attention_runtime():
            pass
