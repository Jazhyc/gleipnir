"""The selected QAT recipe keeps reference arithmetic and restores all scopes."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from gleipnir.training import qat_reference


def test_selected_runtime_requires_installed_qat():
    with pytest.raises(ValueError, match="install"):
        with qat_reference.attention_runtime(SimpleNamespace()):
            pass


def test_selected_runtime_keeps_reference_precision_and_restores(monkeypatch):
    events = []

    @contextmanager
    def scope(*args, **kwargs):
        events.append((args, kwargs))
        try:
            yield
        finally:
            events.append("restored")

    monkeypatch.setattr(qat_reference.quantization_recovery, "attention_runtime", scope)
    monkeypatch.setattr(qat_reference, "optimized_path", scope)
    model = SimpleNamespace(_gleipnir_qad_groups=[])
    with pytest.raises(RuntimeError), qat_reference.attention_runtime(model):
        raise RuntimeError("training failed")
    assert events == [
        ((), {}),
        ((model,), {"fp8": False, "mlp": True, "gdn": True}),
        "restored",
        "restored",
    ]
