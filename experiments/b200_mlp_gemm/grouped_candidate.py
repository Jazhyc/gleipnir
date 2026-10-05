"""Matched native grouped Q/K screen without reloading the resident model."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path

import torch

import gleipnir.grouped_gdn as integration
from experiments.b200_mlp_gemm.grouped_kernel_audit import GroupedKernelAudit
from experiments.b200_mlp_gemm.resident_worker import write_json
from experiments.b200_mlp_gemm.warmed_training import (
    collect_batches,
    physical_contract,
    preserve_random_state,
)
from gleipnir.training_execution_audit import tensor_digest

_CONTEXT = None
_INSTALLATION = None


@contextmanager
def intervention(trainer):
    global _CONTEXT, _INSTALLATION
    _CONTEXT = integration.grouped_gdn_context(trainer.model)
    _INSTALLATION = _CONTEXT.__enter__()
    try:
        yield
    finally:
        profile = getattr(trainer, "_gleipnir_grouped_audit", None)
        if profile is not None:
            trainer.remove_callback(profile)
            del trainer._gleipnir_grouped_audit
        _CONTEXT.__exit__(None, None, None)
        _CONTEXT = None


def validate(trainer) -> dict:
    """Check the changed path once without replaying unchanged startup gates."""
    global _CONTEXT, _INSTALLATION
    trial = Path(trainer.args.output_dir).parent
    for name, path in (
        ("executed_grouped_gdn.py", Path(integration.__file__)),
        ("executed_grouped_candidate.py", Path(__file__)),
        (
            "executed_grouped_kernel_audit.py",
            Path(__file__).with_name("grouped_kernel_audit.py"),
        ),
    ):
        (trial / name).write_bytes(path.read_bytes())
    reference = json.loads((trial.parent / "01baseline/receipt.json").read_text())
    parameters = [p for p in trainer.model.parameters() if p.requires_grad]
    if not parameters or any(p.dtype != torch.float32 for p in parameters):
        raise ValueError("grouped screen requires FP32 master adapters")
    initial = tensor_digest(parameters)
    if trainer.optimizer is not None or trainer.lr_scheduler is not None:
        raise ValueError("grouped validation requires reset optimizer/scheduler")
    saved = {
        name: getattr(trainer, name)
        for name in (
            "microbatch_records",
            "logical_batch_sizes",
            "microbatch_peak_allocated",
            "microbatch_peak_reserved",
            "_microbatch_loss_weight",
        )
    }
    prior_accumulation = getattr(trainer, "current_gradient_accumulation_steps", None)
    started = time.perf_counter()
    try:
        trainer.current_gradient_accumulation_steps = 1
        with preserve_random_state():
            batch = collect_batches(trainer.get_train_dataloader(), 1)[0]
            _CONTEXT.__exit__(None, None, None)
            trainer.microbatch_records, trainer.logical_batch_sizes = [], []
            trainer.model.zero_grad(set_to_none=True)
            baseline_loss = float(
                trainer.training_step(trainer.model_wrapped, batch, None)
            )
            check_gradients(parameters)
            baseline_contract = physical_contract(trainer.microbatch_records)
            baseline_gradients = [p.grad.detach().cpu().clone() for p in parameters]
            _CONTEXT = integration.grouped_gdn_context(trainer.model)
            _INSTALLATION = _CONTEXT.__enter__()
            trainer.microbatch_records, trainer.logical_batch_sizes = [], []
            trainer.model.zero_grad(set_to_none=True)
            print("grouped_parity candidate_start", flush=True)
            candidate_loss = float(
                trainer.training_step(trainer.model_wrapped, batch, None)
            )
            check_gradients(parameters)
            candidate_contract = physical_contract(trainer.microbatch_records)
            error = norm = 0.0
            exact = True
            for p, g in zip(parameters, baseline_gradients, strict=True):
                actual = p.grad.detach().cpu()
                exact = exact and torch.equal(actual, g)
                error += float((actual.double() - g.double()).square().sum())
                norm += float(g.double().square().sum())
            relative = math.sqrt(error / max(norm, 1e-30))
            del baseline_gradients
            expected = [
                r
                for r in reference["physical_contract"]
                if r["update"] == reference["physical_contract"][0]["update"]
            ]
            passed = (
                math.isfinite(relative)
                and math.isfinite(candidate_loss)
                and math.isfinite(baseline_loss)
                and abs(candidate_loss - baseline_loss) <= 0.005
                and relative <= 0.05
                and baseline_contract == candidate_contract == expected
            )
            receipt = {
                "accepted_for_timing": passed,
                "performed_this_trial": True,
                "worker_pid": os.getpid(),
                "initial_master_sha256": initial,
                "installation": _INSTALLATION,
                "baseline_first_batch_loss": baseline_loss,
                "candidate_first_batch_loss": candidate_loss,
                "absolute_loss_difference": abs(candidate_loss - baseline_loss),
                "adapter_gradient_relative_l2": relative,
                "adapter_gradients_bitwise_equal": exact,
                "strict_gradient_limit": 0.05,
                "absolute_loss_limit": 0.005,
                "physical_contract_agreement": baseline_contract == candidate_contract,
                "unchanged_startup_checks_reused": True,
                "full_model_preparation_replayed": False,
                "integration_source_sha256": hashlib.sha256(
                    Path(integration.__file__).read_bytes()
                ).hexdigest(),
                "masters_unchanged": tensor_digest(parameters) == initial,
                "optimizer_updates": 0,
                "validation_wall_seconds": time.perf_counter() - started,
            }
            write_json(trial / "grouped_validation.json", receipt)
            print(f"grouped_parity {json.dumps(receipt)}", flush=True)
            if not passed or not receipt["masters_unchanged"]:
                raise ValueError("grouped first-batch strict parity failed")
            trainer.model.zero_grad(set_to_none=True)
            if tensor_digest(parameters) != initial:
                raise ValueError("grouped validation changed FP32 masters")
            receipt.update(
                masters_unchanged=True,
                optimizer_updates=0,
                validation_wall_seconds=time.perf_counter() - started,
            )
            write_json(trial / "grouped_validation.json", receipt)
        audit = GroupedKernelAudit(trial)
        trainer.add_callback(audit)
        trainer._gleipnir_grouped_audit = audit
        return receipt
    finally:
        trainer.model.zero_grad(set_to_none=True)
        for name, value in saved.items():
            setattr(trainer, name, value)
        if prior_accumulation is None:
            del trainer.current_gradient_accumulation_steps
        else:
            trainer.current_gradient_accumulation_steps = prior_accumulation


def check_gradients(parameters):
    gradients = [p.grad for p in parameters]
    if not gradients or any(g is None for g in gradients):
        raise FloatingPointError("missing grouped-screen adapter gradients")
    torch.nn.utils.get_total_norm(gradients, error_if_nonfinite=True)
