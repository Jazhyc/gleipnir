"""Warmup must preserve the training trajectory and prove compilation stopped."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from experiments.b200_mlp_gemm.warmed_training import (
    collect_batches,
    counter_delta,
    physical_contract,
    validate_warmed_receipt,
    warm_shapes,
)


def test_epoch_collection_preserves_sampler_and_cpu_rng():
    from accelerate.data_loader import DataLoaderShard, SeedableRandomSampler

    dataset = list(range(8))
    sampler = SeedableRandomSampler(dataset, data_seed=41)
    loader = DataLoaderShard(dataset, batch_size=2, sampler=sampler)
    rng = torch.get_rng_state().clone()
    collected = collect_batches(loader, 8)
    assert torch.equal(torch.get_rng_state(), rng)
    assert loader.iteration == sampler.epoch == 0
    assert sampler.generator is None
    actual = []
    for epoch in range(2):
        loader.set_epoch(epoch)
        actual.extend(list(loader))
    assert all(torch.equal(a, b) for a, b in zip(collected, actual, strict=True))
    assert not torch.equal(collected[0], collected[4])


class ToyTrainer:
    def __init__(self, *, invalid=None):
        self.model_wrapped = torch.nn.Linear(3, 1)
        self.optimizer = torch.optim.AdamW(self.model_wrapped.parameters())
        self.state = SimpleNamespace(global_step=0)
        self.microbatch_records = []
        self.logical_batch_sizes = []
        self.microbatch_peak_allocated = self.microbatch_peak_reserved = 0
        self._microbatch_loss_weight = 1.0
        self.seen = set()
        self.plans = 0
        self.invalid = invalid

    def snapshot(self):
        return {"plans": self.plans}

    def training_step(self, model, inputs, num_items):
        if inputs["step"] not in self.seen or self.invalid == "never_warm":
            self.seen.add(inputs["step"])
            self.plans += 1
        self.logical_batch_sizes.append(1)
        self.microbatch_records.append(record(inputs["step"]))
        loss = model(torch.randn(2, 3)).square().mean()
        if self.invalid == "nonfinite":
            loss = loss * float("nan")
        loss.backward()
        if self.invalid == "missing":
            model.bias.grad = None
        if self.invalid == "master":
            with torch.no_grad():
                model.bias.add_(1)
        return loss.detach()


def record(step):
    return {
        "update": step,
        "logical_indices": [0],
        "tokens": step,
        "padded_tokens": step,
    }


def test_warmup_restores_rng_masters_and_training_records(tmp_path):
    trainer = ToyTrainer()
    before = [p.detach().clone() for p in trainer.model_wrapped.parameters()]
    rng = torch.get_rng_state().clone()
    receipt = warm_shapes(
        trainer,
        [{"step": i} for i in range(1, 21)],
        trainer.snapshot,
        [record(i) for i in range(1, 21)],
        tmp_path / "warm.json",
    )
    assert len(receipt["passes"]) == 2
    assert receipt["passes"][0]["delta"] == {"plans": 20}
    assert receipt["passes"][1]["delta"] == {"plans": 0}
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(
        torch.equal(a, b)
        for a, b in zip(before, trainer.model_wrapped.parameters(), strict=True)
    )
    assert all(p.grad is None for p in trainer.model_wrapped.parameters())
    assert not trainer.optimizer.state
    assert trainer.microbatch_records == trainer.logical_batch_sizes == []
    assert not hasattr(trainer, "current_gradient_accumulation_steps")


@pytest.mark.parametrize(
    "invalid,pattern",
    [
        ("never_warm", "still compiles"),
        ("master", "changed adapters"),
        ("missing", "missing warmup"),
        ("nonfinite", "nonfinite warmup"),
    ],
)
def test_warmup_rejects_drift_and_cleans_gradients(tmp_path, invalid, pattern):
    trainer = ToyTrainer(invalid=invalid)
    with pytest.raises((ValueError, FloatingPointError), match=pattern):
        warm_shapes(
            trainer,
            [{"step": i} for i in range(1, 21)],
            trainer.snapshot,
            [record(i) for i in range(1, 21)],
            tmp_path / "warm.json",
        )
    assert all(p.grad is None for p in trainer.model_wrapped.parameters())
    assert trainer.microbatch_records == []


def warm_receipt():
    return {
        "status": "warmup_complete",
        "masters_unchanged": True,
        "optimizer_state_unchanged": True,
        "optimizer_updates": 0,
        "initial_master_sha256": "initial",
        "final_master_sha256": "initial",
        "passes": [{}, {"physical_contract": [], "delta": {"plans": 0}}],
        "update_audit": [{"step": i, "delta": {"plans": 0}} for i in range(1, 21)],
    }


@pytest.mark.parametrize(
    "invalid", ["compile", "plan", "incomplete", "master", "coverage"]
)
def test_warmed_claim_requires_complete_measurement_evidence(invalid):
    receipt = warm_receipt()
    candidate = {"initial_master_sha256": "initial", "physical_contract": []}
    validate_warmed_receipt(receipt, candidate)
    bad = deepcopy(receipt)
    if invalid == "compile":
        bad["update_audit"][10]["delta"]["specializations"] = 1
    elif invalid == "plan":
        bad["update_audit"][19]["delta"]["plans"] = 1
    elif invalid == "incomplete":
        bad["update_audit"].pop()
    elif invalid == "master":
        bad["masters_unchanged"] = False
    else:
        bad["passes"][-1]["physical_contract"] = [record(1)]
    with pytest.raises(ValueError):
        validate_warmed_receipt(bad, candidate)
    assert counter_delta({"plans": 10}, {"plans": 11}) == {"plans": 1}
    assert physical_contract([record(1)]) == [record(1)]
