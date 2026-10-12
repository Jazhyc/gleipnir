"""Freeze attention-only substitutions and the non-mutating gradient diagnostic."""

import json
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

from experiments.native_fp8_qat.generate import transform
from experiments.native_mxfp8_qat import runtime


@pytest.mark.parametrize("stage", ["train", "reference", "evaluate"])
def test_attention_arm_substitutions_are_reversible(stage):
    source = Path(f"experiments/quantization_recovery/{stage}.py").read_text()
    generated, changes = transform(
        source,
        stage,
        config_path="experiments/native_mxfp8_qat/config.yaml",
        runtime_module="experiments.native_mxfp8_qat.runtime",
        native_attention_backward=True,
    )
    restored = generated
    for change in reversed(changes):
        restored = restored.replace(change["new"], change["old"])
    assert restored == source
    if stage == "train":
        assert "validate_model_backward(model, logits, short, output)" in generated
        assert "approximate_low_rank_STE_and_native_MXFP8" in generated


@pytest.mark.parametrize("fail", [False, True])
def test_paired_gradients_preserve_weights_rng_mode_and_clear_grads(
    monkeypatch, tmp_path, fail
):
    model = torch.nn.Linear(2, 2, bias=False).float().eval()
    weights = model.weight.detach().clone()
    state = torch.get_rng_state().clone()
    selected = []

    @contextmanager
    def scope(name):
        selected.append(name)
        try:
            yield
        finally:
            selected.pop()

    monkeypatch.setattr(runtime.control, "attention_runtime", lambda: scope("control"))
    monkeypatch.setattr(runtime, "attention_runtime", lambda: scope("native"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def logits(rows):
        torch.rand(1)
        value = model(torch.ones(len(rows), 2))
        if selected[-1] == "native":
            if fail:
                raise RuntimeError("diagnostic failure")
            value = value.detach() + 1.1 * (value - value.detach())
        return value

    if fail:
        with pytest.raises(RuntimeError, match="diagnostic failure"):
            runtime.validate_model_backward(model, logits, [{"id": 1}], tmp_path)
    else:
        runtime.validate_model_backward(model, logits, [{"id": 1}], tmp_path)
        receipt = json.loads((tmp_path / "whole_model_backward.json").read_text())
        assert receipt["passed"]
        assert receipt["gradient_relative_l2"] == pytest.approx(0.1, abs=1e-6)
    assert torch.equal(model.weight, weights)
    assert torch.equal(torch.get_rng_state(), state)
    assert model.weight.grad is None
    assert not model.training
    assert not selected
