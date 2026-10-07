"""Targeted exact first-batch gate for direct native FROST dispatch."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

import torch

from experiments.b200_mlp_gemm.warmed_training import (
    collect_batches,
    physical_contract,
    preserve_random_state,
)
from gleipnir.kernels.fp4 import frost_bindings
from gleipnir.training_execution_audit import tensor_digest

_CONTROLLER = None


def finish() -> dict:
    """Record whether the real training loop exercised the host path."""
    if _CONTROLLER is None:
        raise ValueError("binding scope already closed")
    return _CONTROLLER.state()


@contextmanager
def intervention(trainer):
    global _CONTROLLER
    with frost_bindings.training_frost_bindings() as controller:
        _CONTROLLER = controller
        try:
            yield
        finally:
            _CONTROLLER = None


def validate(trainer) -> dict:
    """Compare one whole logical batch without updating masters or RNG."""
    controller = _CONTROLLER
    if (
        controller is None
        or trainer.optimizer is not None
        or trainer.lr_scheduler is not None
    ):
        raise ValueError("targeted validation requires a scoped, reset Trainer")
    root = Path(os.environ["GLEIPNIR_FP4_RESIDENT_ROOT"])
    binding_source = Path(frost_bindings.__file__).read_bytes()
    source_receipt = root / "executed_frost_bindings.py"
    if source_receipt.exists() and source_receipt.read_bytes() != binding_source:
        raise ValueError("resident binding implementation changed")
    source_receipt.write_bytes(binding_source)
    reference = json.loads((root / "01baseline/receipt.json").read_text())
    parameters = [p for p in trainer.model.parameters() if p.requires_grad]
    initial = tensor_digest(parameters)
    fields = (
        "microbatch_records",
        "logical_batch_sizes",
        "microbatch_peak_allocated",
        "microbatch_peak_reserved",
        "_microbatch_loss_weight",
    )
    saved = {name: getattr(trainer, name) for name in fields}
    prior_accumulation = getattr(trainer, "current_gradient_accumulation_steps", None)
    try:
        trainer.current_gradient_accumulation_steps = 1
        with preserve_random_state():
            batch = collect_batches(trainer.get_train_dataloader(), 1)[0]
            losses, contracts, digests = [], [], []
            baseline = None
            finite = exact = True
            calls_before = controller.state()["direct_calls"]
            for mode in ("original", "direct"):
                torch.cuda.synchronize()
                controller.set_mode(mode)
                trainer.microbatch_records, trainer.logical_batch_sizes = [], []
                trainer.model.zero_grad(set_to_none=True)
                loss = trainer.training_step(trainer.model_wrapped, batch, None)
                torch.cuda.synchronize()
                losses.append(float(loss))
                contracts.append(physical_contract(trainer.microbatch_records))
                gradients = []
                for p in parameters:
                    if p.grad is None:
                        raise ValueError("missing adapter gradient in targeted gate")
                    g = p.grad.detach().cpu().clone()
                    finite = finite and bool(torch.isfinite(g).all())
                    gradients.append(g)
                digests.append(tensor_digest(gradients))
                if baseline is None:
                    baseline = gradients
                else:
                    exact = all(
                        torch.equal(a, b)
                        for a, b in zip(baseline, gradients, strict=True)
                    )
            expected = [
                r
                for r in reference["physical_contract"]
                if r["update"] == reference["physical_contract"][0]["update"]
            ]
            master_unchanged = tensor_digest(parameters) == initial
            calls = controller.state()["direct_calls"] - calls_before
            passed = (
                finite
                and exact
                and losses[0] == losses[1]
                and contracts[0] == contracts[1] == expected
                and master_unchanged
            )
            return {
                "accepted_for_timing": passed,
                "baseline_first_batch_loss": losses[0],
                "candidate_first_batch_loss": losses[1],
                "adapter_gradients_bitwise_equal": exact,
                "adapter_gradient_sha256": digests,
                "finite_gradients": finite,
                "masters_unchanged": master_unchanged,
                "physical_contract_agreement": contracts[0] == contracts[1] == expected,
                "initial_master_sha256": initial,
                "optimizer_updates": 0,
                "direct_host_calls_in_gate": calls,
                "note": (
                    "Existing CUDA graphs may replay GPU work without host dispatch."
                ),
                "binding_source_sha256": hashlib.sha256(
                    Path(frost_bindings.__file__).read_bytes()
                ).hexdigest(),
                "binding_state": controller.state(),
                "unchanged_startup_checks_reused": True,
            }
    finally:
        trainer.model.zero_grad(set_to_none=True)
        for name, value in saved.items():
            setattr(trainer, name, value)
        if prior_accumulation is None:
            del trainer.current_gradient_accumulation_steps
        else:
            trainer.current_gradient_accumulation_steps = prior_accumulation
        torch.cuda.synchronize()
        controller.set_mode("direct")
