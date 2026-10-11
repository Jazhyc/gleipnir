"""Negative-tail selection and exact logical-gradient replay across partitions."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from gleipnir.training.adaptive_microbatching import (
    AdaptiveMicrobatchTrainerMixin,
    LogicalBatch,
    MicrobatchPolicy,
)
from gleipnir.training.partial_auc import negative_tail_weights, partial_auc_loss

from .train import gradient_canary, install


def test_fractional_mass_and_boundary_ties():
    scores = torch.tensor([5.0, 4.0, 4.0, 1.0, 0.0])
    assert torch.allclose(
        negative_tail_weights(scores, 0.4), torch.tensor([1.0, 0.5, 0.5, 0.0, 0.0])
    )
    assert torch.allclose(
        negative_tail_weights(scores, 0.2), torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
    )
    assert torch.allclose(
        negative_tail_weights(torch.ones(5), 0.2), torch.full((5,), 0.2)
    )


def test_focuses_negative_tail_and_preserves_both_sides_gradients():
    margins = torch.tensor([-5.0, 2.0, 0.0], requires_grad=True)
    labels = torch.tensor([0, 0, 1])
    sources = torch.zeros(3, dtype=torch.long)
    loss = partial_auc_loss(margins, labels, sources, 0.2)
    grad = torch.autograd.grad(loss, margins)[0]
    assert grad[0] == 0 and grad[1] > 0 and grad[2] < 0
    assert torch.allclose(loss, F.softplus(torch.tensor(2.0)))


def test_single_class_source_does_not_add_cross_source_pairs():
    margins = torch.tensor([1.0, -1.0], requires_grad=True)
    loss = partial_auc_loss(margins, torch.tensor([1, 0]), torch.tensor([0, 1]))
    assert loss == 0
    assert torch.equal(torch.autograd.grad(loss, margins)[0], torch.zeros(2))


def test_parameter_gradient_canary():
    assert gradient_canary()["passed"]


def test_real_adaptive_mixin_replay_matches_full_batch_gradient(tmp_path, monkeypatch):
    def ordinary(logits, targets, **kwargs):
        return F.binary_cross_entropy_with_logits(
            logits[:, 1] - logits[:, 0], targets, reduction="none"
        )

    module = SimpleNamespace(soft_binary_distillation_losses=ordinary)

    class Base:
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            loss = module.soft_binary_distillation_losses(
                model(inputs["x"]), inputs["q"]
            ).mean()
            return (loss, {}) if return_outputs else loss

        def training_step(self, model, inputs, num_items_in_batch=None):
            loss = self.compute_loss(model, inputs)
            loss.backward()
            return loss.detach()

        def _prepare_inputs(self, inputs):
            return inputs

        def compute_loss_context_manager(self):
            return nullcontext()

    class Trainer(AdaptiveMicrobatchTrainerMixin, Base):
        pass

    features = [
        {
            "direct_input_ids": [1, 2],
            "x": x,
            "soft_target": q,
            "binary_label": y,
            "dataset_id": 0,
        }
        for x, q, y in [
            ([1.0, 2.0], 0.1, 0),
            ([2.0, 1.0], 0.2, 0),
            ([3.0, 1.0], 0.9, 1),
            ([1.0, 3.0], 0.8, 1),
        ]
    ]

    def collator(rows):
        return {
            "x": torch.tensor([r["x"] for r in rows]),
            "q": torch.tensor([r["soft_target"] for r in rows]),
        }

    trainer = Trainer()
    trainer.args = SimpleNamespace(
        world_size=1, n_gpu=1, gradient_accumulation_steps=1, device=torch.device("cpu")
    )
    trainer.model_accepts_loss_kwargs = False
    trainer.compute_loss_func = None
    trainer.eval_dataset = None
    trainer.current_gradient_accumulation_steps = 1
    trainer.enable_adaptive_microbatching(
        MicrobatchPolicy(2, 1), collator, profile=False, sequence_packing=True
    )
    model = torch.nn.Linear(2, 2, bias=False)
    batch = collator(features)
    logits = model(batch["x"])
    expected_loss = ordinary(logits, batch["q"]).mean() + 0.1 * partial_auc_loss(
        logits[:, 1] - logits[:, 0],
        torch.tensor([0, 0, 1, 1]),
        torch.zeros(4, dtype=torch.long),
    )
    expected = torch.autograd.grad(expected_loss, model.weight)[0]
    monkeypatch.setattr(
        AdaptiveMicrobatchTrainerMixin,
        "training_step",
        AdaptiveMicrobatchTrainerMixin.training_step,
    )
    install(
        module,
        {"weight": 0.1, "beta": 0.2, "temperature": 1},
        tmp_path / "receipt.json",
    )
    value = trainer.training_step(model, {"logical_batch": LogicalBatch(features)})
    assert torch.allclose(model.weight.grad, expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(value, expected_loss.detach(), atol=1e-6, rtol=1e-6)
    assert trainer.logical_batch_sizes == [4]


@pytest.mark.parametrize("beta", [0.0, -0.1, 1.1, float("nan")])
def test_invalid_tail_fraction(beta):
    with pytest.raises(ValueError):
        negative_tail_weights(torch.ones(2), beta)


def test_launcher_uses_persistent_offline_cache_and_pinned_path(tmp_path, monkeypatch):
    from . import run

    captured = {}
    monkeypatch.setattr(run, "training_environment", lambda *a, **kw: {"PATH": "/old"})

    def start(command, **kwargs):
        captured.update(command=command, **kwargs)
        return SimpleNamespace(wait=lambda timeout: 0)

    monkeypatch.setattr(run.subprocess, "Popen", start)
    ctx = SimpleNamespace(
        config={"training_python": "/tmp/pinned/bin/python"}, logs=tmp_path / "logs"
    )
    run.launch(ctx, "train")
    env = captured["env"]
    assert env["HF_HOME"] == str(run.ROOT / ".cache/huggingface")
    assert env["HF_HUB_CACHE"] == str(run.ROOT / ".cache/huggingface/hub")
    assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["PATH"].startswith("/tmp/pinned/bin:/usr/local/cuda/bin:")
