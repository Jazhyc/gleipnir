"""The targeted full-model gate must reject changed loss/gradients and restore state."""

import json
from types import SimpleNamespace

import pytest
import torch

from experiments.b200_frost_training import candidate


@pytest.mark.parametrize("difference", ["none", "loss", "gradient", "missing"])
def test_targeted_gate_restores_trainer_and_rejects_drift(
    monkeypatch, tmp_path, difference
):
    record = {"update": 1, "logical_indices": [0], "tokens": 14, "padded_tokens": 14}
    (tmp_path / "01baseline").mkdir()
    (tmp_path / "01baseline/receipt.json").write_text(
        json.dumps({"physical_contract": [record]})
    )
    monkeypatch.setenv("GLEIPNIR_FP4_RESIDENT_ROOT", str(tmp_path))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(candidate, "collect_batches", lambda loader, count: [object()])
    controller = SimpleNamespace(mode="direct")
    controller.set_mode = lambda mode: setattr(controller, "mode", mode)
    controller.state = lambda: {"direct_calls": 0, "mode": controller.mode}
    monkeypatch.setattr(candidate, "_CONTROLLER", controller)
    model = torch.nn.Linear(2, 1)
    records, sizes = [], []
    trainer = SimpleNamespace(
        model=model,
        model_wrapped=model,
        optimizer=None,
        lr_scheduler=None,
        microbatch_records=records,
        logical_batch_sizes=sizes,
        microbatch_peak_allocated=0,
        microbatch_peak_reserved=0,
        _microbatch_loss_weight=1.0,
        get_train_dataloader=lambda: object(),
    )

    def training_step(*args):
        trainer.microbatch_records.append(record.copy())
        for p in model.parameters():
            p.grad = torch.ones_like(p)
            if controller.mode == "direct" and difference == "gradient":
                p.grad.mul_(2)
            if controller.mode == "direct" and difference == "missing":
                p.grad = None
        return torch.tensor(
            2.0 if controller.mode == "direct" and difference == "loss" else 1.0
        )

    trainer.training_step = training_step
    initial = [p.detach().clone() for p in model.parameters()]
    if difference == "missing":
        with pytest.raises(ValueError, match="missing adapter gradient"):
            candidate.validate(trainer)
    else:
        receipt = candidate.validate(trainer)
        assert receipt["accepted_for_timing"] is (difference == "none")
    assert trainer.microbatch_records is records
    assert trainer.logical_batch_sizes is sizes
    assert not hasattr(trainer, "current_gradient_accumulation_steps")
    assert controller.mode == "direct"
    for p, original in zip(model.parameters(), initial, strict=True):
        assert p.grad is None
        assert torch.equal(p, original)
