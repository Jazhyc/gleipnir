"""Matched resident screens for CPU packing metadata and normalization copies."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path

import torch

import gleipnir.training_hotpath as integration
from experiments.b200_mlp_gemm import hotpath_reuse
from experiments.b200_mlp_gemm.hotpath_reuse import reusable_validation
from experiments.b200_mlp_gemm.resident_worker import write_json
from experiments.b200_mlp_gemm.warmed_training import (
    collect_batches,
    physical_contract,
    preserve_random_state,
)
from gleipnir.training_execution_audit import tensor_digest

NORMALIZE = False
MATCH_NORM_CONFIGS = True
PREPARED_METADATA = False
ASYNC_INPUTS = True
PROFILE = False
_CONTEXT = None
_INSTALLATION = None


@contextmanager
def installed(trainer):
    with ExitStack() as stack:
        metadata = stack.enter_context(
            integration.training_hotpath_context(
                normalize=NORMALIZE,
                match_norm_configs=MATCH_NORM_CONFIGS,
                prepare_metadata=PREPARED_METADATA,
            )
        )
        if ASYNC_INPUTS:
            stack.enter_context(integration.nonblocking_trainer_inputs(trainer))
        metadata["nonblocking_trainer_inputs"] = ASYNC_INPUTS
        yield metadata


@contextmanager
def intervention(trainer):
    global _CONTEXT, _INSTALLATION
    _CONTEXT = installed(trainer)
    _INSTALLATION = _CONTEXT.__enter__()
    try:
        yield
    finally:
        profile = getattr(trainer, "_gleipnir_hotpath_profile", None)
        if profile is not None:
            profile.close()
            trainer.remove_callback(profile)
            del trainer._gleipnir_hotpath_profile
        _CONTEXT.__exit__(None, None, None)
        _CONTEXT = None


def validate(trainer) -> dict:
    """Check the changed path once without replaying unchanged startup gates."""
    global _CONTEXT, _INSTALLATION
    trial = Path(trainer.args.output_dir).parent
    for name, path in (
        ("executed_training_hotpath.py", Path(integration.__file__)),
        ("executed_hotpath_candidate.py", Path(__file__)),
        ("executed_hotpath_reuse.py", Path(hotpath_reuse.__file__)),
    ):
        (trial / name).write_bytes(path.read_bytes())
    reference = json.loads((trial.parent / "01baseline/receipt.json").read_text())
    parameters = [p for p in trainer.model.parameters() if p.requires_grad]
    initial = tensor_digest(parameters)
    if trainer.optimizer is not None or trainer.lr_scheduler is not None:
        raise ValueError("hotpath validation requires reset optimizer/scheduler")
    reused = reusable_validation(
        trial.parent,
        integration_sha256=hashlib.sha256(
            Path(integration.__file__).read_bytes()
        ).hexdigest(),
        installer_source=Path(__file__).read_text(),
        normalize=NORMALIZE,
        async_inputs=ASYNC_INPUTS,
        prepared_metadata=PREPARED_METADATA,
        worker_pid=os.getpid(),
        initial_master=initial,
        physical_contract=reference["physical_contract"],
    )
    if reused is not None and not PROFILE:
        write_json(trial / "hotpath_validation.json", reused)
        print(f"hotpath_validation reused={reused['reuse_reference']}", flush=True)
        return reused
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
    profile = None
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
            baseline_contract = physical_contract(trainer.microbatch_records)
            baseline_gradients = [p.grad.detach().cpu().clone() for p in parameters]
            _CONTEXT = installed(trainer)
            _INSTALLATION = _CONTEXT.__enter__()
            trainer.microbatch_records, trainer.logical_batch_sizes = [], []
            trainer.model.zero_grad(set_to_none=True)
            candidate_loss = float(
                trainer.training_step(trainer.model_wrapped, batch, None)
            )
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
            }
            write_json(trial / "hotpath_validation.json", receipt)
            print(f"hotpath_parity {json.dumps(receipt)}", flush=True)
            if not passed:
                raise ValueError("hotpath first-batch strict parity failed")
            trainer.model.zero_grad(set_to_none=True)
            preparation = []
            if NORMALIZE:
                # Only the existing normalization kernel gets a new input dtype.
                # Its autotune keys are D=128 and ceil(tokens*32/65536), not T.
                bins = {}
                for row in reference["physical_contract"]:
                    tokens = row["tokens"]
                    bins.setdefault((tokens * 32 + 65535) // 65536, tokens)
                for nb, tokens in sorted(bins.items()):
                    x = torch.zeros(
                        (1, tokens, 32, 128),
                        dtype=torch.bfloat16,
                        device=parameters[0].device,
                        requires_grad=True,
                    )
                    y = integration.normalize_qk_without_input_copy(x)
                    y.sum().backward()
                    torch.cuda.synchronize()
                    preparation.append(
                        {"normalization_autotune_nb": nb, "tokens": tokens}
                    )
                    del x, y
                receipt["normalization_preparation"] = preparation
            if tensor_digest(parameters) != initial:
                raise ValueError("hotpath validation changed FP32 masters")
            receipt.update(
                masters_unchanged=True,
                optimizer_updates=0,
                validation_wall_seconds=time.perf_counter() - started,
            )
            write_json(trial / "hotpath_validation.json", receipt)
        if PROFILE:
            from experiments.b200_mlp_gemm.resident_profile import GemmProfile

            profile = GemmProfile(trial)
            trainer.add_callback(profile)
            # Scoped intervention cleanup removes this callback after training.
            trainer._gleipnir_hotpath_profile = profile
        return receipt
    finally:
        trainer.model.zero_grad(set_to_none=True)
        for name, value in saved.items():
            setattr(trainer, name, value)
        if prior_accumulation is None:
            del trainer.current_gradient_accumulation_steps
        else:
            trainer.current_gradient_accumulation_steps = prior_accumulation
