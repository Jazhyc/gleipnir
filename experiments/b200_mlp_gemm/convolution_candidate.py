"""Hot-loaded NVIDIA BF16 causal Conv1D trial on the resident FP4 MLP baseline."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import TrainerCallback

import gleipnir.nvidia_causal_conv1d as integration
from experiments.b200_mlp_gemm import convolution_reuse
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
from gleipnir.training_execution_audit import tensor_digest

_CONTEXT = None
_INSTALLATION = None
_MODEL = None
RETURN_REFERENCE_DIAGNOSTIC = False
TIMING_AFTER_FAILED_PARITY = True


def relative_l2(actual, reference):
    return float(
        (actual.float() - reference.float()).norm()
        / reference.float().norm().clamp_min(1e-30)
    )


def isolated_canary(backend):
    """Independent FP32 convolution, Dao BF16 parity and packed isolation."""
    from causal_conv1d import causal_conv1d_fn

    lengths = (1, 2, 3, 63, 65, 256, 1025)
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    cu = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    ids = torch.repeat_interleave(
        torch.arange(len(lengths), device="cuda", dtype=torch.int32),
        torch.tensor(lengths, device="cuda"),
    ).unsqueeze(0)
    x = (
        torch.randn(1, offsets[-1], 8192, device="cuda", dtype=torch.bfloat16)
        .transpose(1, 2)
        .requires_grad_()
    )
    weight = (
        torch.randn(8192, 4, device="cuda", dtype=torch.bfloat16) * 0.1
    ).contiguous()
    upstream = torch.randn_like(x)
    actual = backend.causal_conv1d(x, weight, activation="silu", cu_seqlens=cu)
    assert backend._get_causal_conv1d_last_route() == "native-autograd"
    dx = torch.autograd.grad(actual, x, upstream)[0]
    dao = causal_conv1d_fn(x, weight, activation="silu", seq_idx=ids)
    dao_dx = torch.autograd.grad(dao, x, upstream)[0]
    xf = x.detach().float().requires_grad_()
    reference = torch.cat(
        [
            F.silu(
                F.conv1d(
                    xf[:, :, a:b], weight.float().unsqueeze(1), padding=3, groups=8192
                )[:, :, : b - a]
            )
            for a, b in zip(offsets[:-1], offsets[1:], strict=True)
        ],
        dim=-1,
    )
    reference_dx = torch.autograd.grad(reference, xf, upstream.float())[0]
    changed = x.detach().clone()
    changed[:, :, : offsets[-2]] += 4
    changed.requires_grad_()
    other = backend.causal_conv1d(changed, weight, activation="silu", cu_seqlens=cu)
    leakage = float(
        (other[:, :, offsets[-2] :] - actual[:, :, offsets[-2] :]).abs().max()
    )
    last_only = torch.zeros_like(upstream)
    last_only[:, :, offsets[-2] :] = upstream[:, :, offsets[-2] :]
    independent_dx = torch.autograd.grad(other, changed, last_only)[0]
    gradient_leakage = float(independent_dx[:, :, : offsets[-2]].abs().max())
    errors = {
        "output_relative_l2_vs_dao": relative_l2(actual, dao),
        "input_gradient_relative_l2_vs_dao": relative_l2(dx, dao_dx),
        "output_relative_l2_vs_fp32": relative_l2(actual, reference),
        "input_gradient_relative_l2_vs_fp32": relative_l2(dx, reference_dx),
        "output_cross_example_leakage": leakage,
        "gradient_cross_example_leakage": gradient_leakage,
    }
    finite = all(
        torch.isfinite(t).all().item() for t in (actual, dx, reference, reference_dx)
    )
    accepted = (
        finite
        and leakage == 0
        and gradient_leakage == 0
        and all(errors[k] <= 0.03 for k in errors if "relative_l2" in k)
    )
    return {"accepted": accepted, "finite": finite, "lengths": lengths, **errors}


@contextmanager
def intervention(trainer):
    global _CONTEXT, _INSTALLATION, _MODEL
    _MODEL = trainer.model
    _CONTEXT = integration.nvidia_convolution_context(_MODEL)
    _INSTALLATION = _CONTEXT.__enter__()
    audit = []
    trial = Path(trainer.args.output_dir).parent

    class Audit(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            self.before = dict(_INSTALLATION["stats"])

        def on_step_end(self, args, state, control, **kwargs):
            delta = {
                key: value - self.before.get(key, 0)
                for key, value in _INSTALLATION["stats"].items()
            }
            audit.append({"step": state.global_step, "delta": delta})
            write_json(trial / "convolution_update_audit.json", {"updates": audit})
            native_calls = delta.get("native-autograd", 0) + delta.get(
                "native-inference", 0
            )
            # The first GDN consumes frozen embeddings/projections and does not
            # require an input gradient. Later layers must retain autograd.
            if native_calls != 24 * sum(
                row["update"] == state.global_step for row in trainer.microbatch_records
            ):
                raise ValueError("not every GDN convolution used a native route")
            if state.global_step > 10 and any(
                delta[k] for k in ("training_compiles", "forward_compiles")
            ):
                raise ValueError("measured convolution updates are not warmed")

    callback = Audit()
    trainer.add_callback(callback)
    try:
        yield
    finally:
        trainer.remove_callback(callback)
        _CONTEXT.__exit__(None, None, None)
        _CONTEXT = None


def validate(trainer):
    """Check changed arithmetic once and prepare each actual benchmark shape."""
    global _CONTEXT, _INSTALLATION
    trial = Path(trainer.args.output_dir).parent
    sources = {}
    for label, file in (
        ("integration", Path(integration.__file__)),
        ("candidate", Path(__file__)),
        ("reuse", Path(convolution_reuse.__file__)),
    ):
        content = file.read_bytes()
        (trial / f"executed_convolution_{label}.py").write_bytes(content)
        sources[label] = hashlib.sha256(content).hexdigest()
    backend = _INSTALLATION["backend"]
    native_root = Path(backend.__file__).parents[1]
    native_sources = [
        Path(backend.__file__),
        native_root / "_causal_conv1d_arch.py",
        *sorted((native_root / "causal_conv1d_bulk_sm100").rglob("*.py")),
    ]
    for file in native_sources:
        target = trial / "native_sources" / file.relative_to(native_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(file.read_bytes())
        sources[str(file.relative_to(native_root))] = hashlib.sha256(
            file.read_bytes()
        ).hexdigest()
    parameters = [p for p in trainer.model.parameters() if p.requires_grad]
    initial = tensor_digest(parameters)
    reference = json.loads((trial.parent / "01baseline/receipt.json").read_text())
    preparation_reference = trial.parent / "09convfinite/candidate_validation.json"
    if (
        TIMING_AFTER_FAILED_PARITY
        and preparation_reference.exists()
        and preparation_reference.parent != trial
    ):
        prior_validation = json.loads(preparation_reference.read_text())
        prior_failure = json.loads(
            (preparation_reference.parent / "failure.json").read_text()
        )
        if convolution_reuse.can_reuse_preparation(
            prior_validation,
            prior_failure,
            sources,
            initial_master=initial,
            worker_pid=os.getpid(),
            baseline_pid=reference["pid"],
        ):
            return {
                **prior_validation,
                "performed_this_trial": False,
                "preparation_reuse_reference": str(
                    preparation_reference.relative_to(trial.parent)
                ),
                "preparation_reuse_sha256": hashlib.sha256(
                    preparation_reference.read_bytes()
                ).hexdigest(),
                "source_sha256": sources,
                "steps": [],
                "preparation_wall_seconds": 0.0,
            }
    saved = {
        n: getattr(trainer, n)
        for n in (
            "microbatch_records",
            "logical_batch_sizes",
            "microbatch_peak_allocated",
            "microbatch_peak_reserved",
            "_microbatch_loss_weight",
        )
    }
    prior = getattr(trainer, "current_gradient_accumulation_steps", None)
    started = time.perf_counter()
    steps = []
    try:
        trainer.current_gradient_accumulation_steps = 1
        with preserve_random_state():
            canary = None
            for previous in sorted(trial.parent.glob("*/convolution_canary.json")):
                archived = previous.parent / "executed_convolution_candidate.py"
                if (
                    previous.parent == trial
                    or not archived.exists()
                    or reference["pid"] != os.getpid()
                ):
                    continue
                old_source = archived.read_text()
                old_function = next(
                    n
                    for n in ast.parse(old_source).body
                    if isinstance(n, ast.FunctionDef) and n.name == "isolated_canary"
                )
                unchanged = (
                    ast.get_source_segment(old_source, old_function)
                    == inspect.getsource(isolated_canary).rstrip()
                )
                unchanged = unchanged and all(
                    (
                        previous.parent
                        / "native_sources"
                        / file.relative_to(native_root)
                    ).exists()
                    and (
                        previous.parent
                        / "native_sources"
                        / file.relative_to(native_root)
                    ).read_bytes()
                    == file.read_bytes()
                    for file in native_sources
                )
                prior_canary = json.loads(previous.read_text())
                if unchanged and prior_canary.get("accepted") is True:
                    canary = {
                        **prior_canary,
                        "performed_this_trial": False,
                        "reuse_reference": str(previous.relative_to(trial.parent)),
                        "reuse_sha256": hashlib.sha256(
                            previous.read_bytes()
                        ).hexdigest(),
                    }
                    break
            if canary is None:
                canary = {**isolated_canary(backend), "performed_this_trial": True}
            write_json(trial / "convolution_canary.json", canary)
            if not canary["accepted"]:
                raise ValueError(f"NVIDIA convolution canary failed: {canary}")
            batches = collect_batches(trainer.get_train_dataloader(), 20)
            parity_reuse = None
            if TIMING_AFTER_FAILED_PARITY:
                prior_trial = trial.parent / "07convdiagnostic"
                parity_path = prior_trial / "convolution_model_parity.json"
                parity_reuse = json.loads(parity_path.read_text())
                failure = json.loads((prior_trial / "failure.json").read_text())
                if (
                    reference["pid"] != os.getpid()
                    or reference["initial_master_sha256"] != initial
                    or failure["initial_master_sha256"] != initial
                    or parity_reuse["accepted"] is not False
                    or canary["accepted"] is not True
                    or not all(
                        (
                            prior_trial
                            / "native_sources"
                            / file.relative_to(native_root)
                        ).read_bytes()
                        == file.read_bytes()
                        for file in native_sources
                    )
                ):
                    raise ValueError("failed-parity timing reference identity changed")
                parity_reuse = {
                    **parity_reuse,
                    "performed_this_trial": False,
                    "reuse_reference": str(parity_path.relative_to(trial.parent)),
                    "reuse_sha256": hashlib.sha256(
                        parity_path.read_bytes()
                    ).hexdigest(),
                }
                write_json(trial / "convolution_model_parity.json", parity_reuse)
                baseline_loss = parity_reuse["baseline_loss"]
                candidate_loss = parity_reuse["candidate_loss"]
                relative = parity_reuse["gradient_relative_l2"]
            else:
                _CONTEXT.__exit__(None, None, None)
                trainer.microbatch_records, trainer.logical_batch_sizes = [], []
                trainer.model.zero_grad(set_to_none=True)
                rng = capture_rng()
                baseline_loss = float(
                    trainer.training_step(trainer.model_wrapped, batches[0], None)
                )
                gradients = [p.grad.detach().float().cpu().clone() for p in parameters]
                _CONTEXT = integration.nvidia_convolution_context(
                    _MODEL, diagnose=True, return_reference=RETURN_REFERENCE_DIAGNOSTIC
                )
                _INSTALLATION = _CONTEXT.__enter__()
                restore_rng(rng)
            stats = _INSTALLATION["stats"]
            trainer.microbatch_records, trainer.logical_batch_sizes = [], []
            for step, batch in enumerate(batches, 1):
                trainer.model.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                before = time.perf_counter()
                loss = float(trainer.training_step(trainer.model_wrapped, batch, None))
                torch.cuda.synchronize()
                if not math.isfinite(loss):
                    raise FloatingPointError("nonfinite convolution preparation loss")
                if step == 1 and TIMING_AFTER_FAILED_PARITY:
                    if abs(loss - candidate_loss) > 1e-7:
                        raise ValueError(
                            "timing trial did not reproduce native first-batch loss"
                        )
                if step == 1 and not TIMING_AFTER_FAILED_PARITY:
                    candidate_loss = loss
                    error = sum(
                        float((p.grad.float().cpu() - g).square().sum())
                        for p, g in zip(parameters, gradients, strict=True)
                    )
                    norm = sum(float(g.square().sum()) for g in gradients)
                    relative = math.sqrt(error / max(norm, 1e-30))
                    del gradients
                    agreement = (
                        abs(candidate_loss - baseline_loss) <= 0.005
                        and relative <= 0.05
                    )
                    write_json(
                        trial / "convolution_model_parity.json",
                        {
                            "accepted": agreement,
                            "baseline_loss": baseline_loss,
                            "candidate_loss": candidate_loss,
                            "gradient_relative_l2": relative,
                        },
                    )
                    write_json(
                        trial / "convolution_output_diagnostics.json",
                        {
                            "convolutions": _INSTALLATION["output_diagnostics"],
                            "returned_dao_output_and_gradients": (
                                RETURN_REFERENCE_DIAGNOSTIC
                            ),
                        },
                    )
                    if RETURN_REFERENCE_DIAGNOSTIC:
                        raise ValueError(
                            "completed wrapper-only Dao passthrough diagnostic; "
                            "no updates requested"
                        )
                    if not agreement:
                        raise ValueError(
                            "BF16 convolution whole-model loss/gradient parity failed"
                        )
                    _CONTEXT.__exit__(None, None, None)
                    _CONTEXT = integration.nvidia_convolution_context(_MODEL)
                    _INSTALLATION = _CONTEXT.__enter__()
                    stats = _INSTALLATION["stats"]
                steps.append({"step": step, "seconds": time.perf_counter() - before})
                write_json(
                    trial / "convolution_preparation.json",
                    {"status": "preparing", "steps": steps, "native": dict(stats)},
                )
                print(
                    f"convolution_prepare step={step}/20 "
                    f"seconds={steps[-1]['seconds']:.3f}",
                    flush=True,
                )
                if time.perf_counter() - started > 1800:
                    raise TimeoutError(
                        "convolution preparation exceeded thirty minutes"
                    )
            if (
                physical_contract(trainer.microbatch_records)
                != reference["physical_contract"]
            ):
                raise ValueError("convolution preparation changed physical partitions")
        if tensor_digest(parameters) != initial:
            raise ValueError("convolution validation changed master adapters")
        return {
            "accepted_for_timing": True,
            "acceptance": "finite_timing_only_after_failed_model_parity"
            if TIMING_AFTER_FAILED_PARITY
            else "changed_kernel_parity",
            "quality_selection_eligible": not TIMING_AFTER_FAILED_PARITY,
            "failed_parity_reference": parity_reuse,
            "performed_this_trial": True,
            "canary": canary,
            "baseline_first_batch_loss": baseline_loss,
            "candidate_first_batch_loss": candidate_loss,
            "adapter_gradient_relative_l2": relative,
            "source_sha256": sources,
            "initial_master_sha256": initial,
            "masters_unchanged": True,
            "optimizer_updates": 0,
            "steps": steps,
            "preparation_wall_seconds": time.perf_counter() - started,
            "precision": "bf16",
            "packing": "existing_cumulative_offsets",
            "native_cache_capacities": {"training": 256, "forward": 256},
        }
    finally:
        trainer.model.zero_grad(set_to_none=True)
        for name, value in saved.items():
            setattr(trainer, name, value)
        if prior is None:
            del trainer.current_gradient_accumulation_steps
        else:
            trainer.current_gradient_accumulation_steps = prior
