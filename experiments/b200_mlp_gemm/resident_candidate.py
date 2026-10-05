"""Hot-loaded GDN FP4 screen on the existing resident FP4 MLP baseline."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path

import torch

import gleipnir.cudnn_fp4_gdn as gdn_source
from experiments.b200_mlp_gemm import candidate_reuse
from experiments.b200_mlp_gemm.resident_worker import (
    capture_rng,
    restore_rng,
    write_json,
)
from experiments.b200_mlp_gemm.warmed_training import (
    collect_batches,
    physical_contract,
    preserve_random_state,
)
from gleipnir.cudnn_fp4_gdn import gdn_fp4_context
from gleipnir.cudnn_fp4_mlp import cache_metadata
from gleipnir.training_execution_audit import tensor_digest

MERGED_INPUTS = False
_INSTALLATION = None
_MODEL = None
_CONTEXT = None


@contextmanager
def intervention(trainer):
    global _INSTALLATION, _MODEL, _CONTEXT
    _MODEL = trainer.model
    _CONTEXT = gdn_fp4_context(_MODEL, merged_inputs=MERGED_INPUTS)
    _INSTALLATION = _CONTEXT.__enter__()
    try:
        yield
    finally:
        _CONTEXT.__exit__(None, None, None)
        _CONTEXT = None


def validate(trainer) -> dict:
    """One matched loss/gradient diagnostic and one new-shape preparation pass."""
    global _CONTEXT, _INSTALLATION
    trial = Path(trainer.args.output_dir).parent
    source = Path(gdn_source.__file__).read_bytes()
    (trial / "executed_cudnn_fp4_gdn.py").write_bytes(source)
    reference = json.loads((trial.parent / "01baseline/receipt.json").read_text())
    parameters = [p for p in trainer.model.parameters() if p.requires_grad]
    initial = tensor_digest(parameters)
    if trainer.optimizer is not None or trainer.lr_scheduler is not None:
        raise ValueError(
            "candidate validation requires reset optimizer/scheduler state"
        )
    reuse_source = Path(candidate_reuse.__file__).read_bytes()
    (trial / "executed_candidate_reuse.py").write_bytes(reuse_source)
    reusable = candidate_reuse.reusable_validation(
        trial.parent,
        integration_sha256=hashlib.sha256(source).hexdigest(),
        merged_inputs=MERGED_INPUTS,
        worker_pid=os.getpid(),
        initial_master=initial,
        physical_contract=reference["physical_contract"],
    )
    if reusable is not None:
        print(f"gdn_fp4_validation reused={reusable['path']}", flush=True)
        return {
            **reusable["validation"],
            "performed_this_trial": False,
            "reuse_reference": reusable["path"],
            "reuse_reference_sha256": reusable["sha256"],
            "steps": [],
            "preparation_wall_seconds": 0.0,
            "installation": _INSTALLATION,
            "reuse_source_sha256": hashlib.sha256(reuse_source).hexdigest(),
        }
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
    steps = []
    started = time.perf_counter()
    try:
        trainer.current_gradient_accumulation_steps = 1
        with preserve_random_state():
            batches = collect_batches(trainer.get_train_dataloader(), 20)
            # Temporarily restore the established FP4-MLP/BF16-GDN recipe.
            _CONTEXT.__exit__(None, None, None)
            trainer.microbatch_records, trainer.logical_batch_sizes = [], []
            trainer.model.zero_grad(set_to_none=True)
            before_comparison = capture_rng()
            baseline_loss = float(
                trainer.training_step(trainer.model_wrapped, batches[0], None)
            )
            baseline_gradients = [
                p.grad.detach().float().cpu().clone() for p in parameters
            ]
            _CONTEXT = gdn_fp4_context(_MODEL, merged_inputs=MERGED_INPUTS)
            _INSTALLATION = _CONTEXT.__enter__()
            restore_rng(before_comparison)
            trainer.microbatch_records, trainer.logical_batch_sizes = [], []
            for step, inputs in enumerate(batches, 1):
                trainer.model.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                before = time.perf_counter()
                loss = float(trainer.training_step(trainer.model_wrapped, inputs, None))
                torch.cuda.synchronize()
                if not math.isfinite(loss):
                    raise FloatingPointError("nonfinite GDN FP4 preparation loss")
                if step == 1:
                    candidate_loss = loss
                    error = sum(
                        float((p.grad.float().cpu() - g).square().sum())
                        for p, g in zip(parameters, baseline_gradients, strict=True)
                    )
                    norm = sum(float(g.square().sum()) for g in baseline_gradients)
                    relative = math.sqrt(error / max(norm, 1e-30))
                    del baseline_gradients
                steps.append({"step": step, "seconds": time.perf_counter() - before})
                write_json(
                    trial / "gdn_preparation.json",
                    {
                        "status": "preparing",
                        "steps": steps,
                        "baseline_first_batch_loss": baseline_loss,
                        "candidate_first_batch_loss": candidate_loss,
                        "adapter_gradient_relative_l2": relative,
                    },
                )
                print(
                    f"gdn_fp4_prepare step={step}/20 "
                    f"seconds={steps[-1]['seconds']:.3f}",
                    flush=True,
                )
                if time.perf_counter() - started > 1800:
                    raise TimeoutError("GDN preparation exceeded thirty minutes")
            contract = physical_contract(trainer.microbatch_records)
            if contract != reference["physical_contract"]:
                raise ValueError("GDN FP4 preparation changed physical batches")
        if tensor_digest(parameters) != initial:
            raise ValueError("GDN FP4 validation changed adapter masters")
        receipt = {
            "accepted_for_timing": True,
            "performed_this_trial": True,
            "acceptance": "selected_finite_timing_only",
            "authority": "2026-10-05 user: Alright, sure try this out.",
            "baseline_first_batch_loss": baseline_loss,
            "candidate_first_batch_loss": candidate_loss,
            "absolute_loss_difference": abs(candidate_loss - baseline_loss),
            "adapter_gradient_relative_l2": relative,
            "numerical_equivalence_claimed": False,
            "initial_master_sha256": initial,
            "masters_unchanged": True,
            "optimizer_updates": 0,
            "steps": steps,
            "preparation_wall_seconds": time.perf_counter() - started,
            "native_cache": cache_metadata(),
            "installation": _INSTALLATION,
            "integration_source_sha256": hashlib.sha256(source).hexdigest(),
            "reuse_source_sha256": hashlib.sha256(reuse_source).hexdigest(),
        }
        write_json(trial / "gdn_preparation.json", {"status": "complete", **receipt})
        print(
            f"gdn_fp4_validation "
            f"loss_difference={receipt['absolute_loss_difference']:.6g} "
            f"gradient_relative_l2={relative:.6g}",
            flush=True,
        )
        return receipt
    finally:
        trainer.model.zero_grad(set_to_none=True)
        for name, value in saved.items():
            setattr(trainer, name, value)
        if prior_accumulation is None:
            del trainer.current_gradient_accumulation_steps
        else:
            trainer.current_gradient_accumulation_steps = prior_accumulation
