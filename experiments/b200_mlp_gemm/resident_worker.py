"""Keep a validated FP4 Trainer resident for bounded, independently reset trials."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import TrainerCallback, TrainerControl

from experiments.b200_mlp_gemm.warmed_training import (
    collect_batches,
    counter_delta,
    physical_contract,
    preserve_random_state,
)
from gleipnir.cudnn_fp4_mlp import cache_metadata
from gleipnir.training_execution_audit import tensor_digest


def write_json(path: Path, value: dict) -> None:
    """Publish receipts atomically so queue readers never see partial JSON."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_request(request: dict) -> None:
    """Requests select bounded known conditions; no arbitrary code execution."""
    if not isinstance(request.get("id"), str) or not request["id"].isalnum():
        raise ValueError("trial id must be alphanumeric")
    if request.get("variant", "baseline") not in {
        "baseline",
        "gemmprofile",
        "candidate",
    }:
        raise ValueError("unsupported resident variant")
    if request.get("variant") == "candidate" and (
        not isinstance(request.get("source_sha256"), str)
        or len(request["source_sha256"]) != 64
    ):
        raise ValueError("candidate source must be checksum-bound")


@contextmanager
def candidate_context(trainer: Any, request: dict, trial: Path):
    """Reload one experiment-owned intervention file while retaining the model."""
    import types

    source_path = Path(__file__).with_name("resident_candidate.py")
    source = source_path.read_bytes()
    if hashlib.sha256(source).hexdigest() != request["source_sha256"]:
        raise ValueError("candidate source checksum drift")
    (trial / "candidate_source.py").write_bytes(source)
    module = types.ModuleType("experiments.b200_mlp_gemm.resident_candidate")
    module.__file__ = str(source_path)
    module.__package__ = "experiments.b200_mlp_gemm"
    exec(compile(source, str(source_path), "exec"), module.__dict__)
    with module.intervention(trainer):
        validation = module.validate(trainer)
        if (
            not isinstance(validation, dict)
            or validation.get("accepted_for_timing") is not True
        ):
            raise ValueError("candidate has no accepted targeted timing validation")
        write_json(trial / "candidate_validation.json", validation)
        yield


def capture_rng() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def reset_trainer(trainer: Any, initial: list[torch.Tensor], rng: dict) -> str:
    """Keep parameter identities and compiled model; clear training state."""
    parameters = [p for p in trainer.model.parameters() if p.requires_grad]
    with torch.no_grad():
        for parameter, value in zip(parameters, initial, strict=True):
            parameter.copy_(value)
    trainer.model.zero_grad(set_to_none=True)
    trainer.optimizer = trainer.lr_scheduler = None
    trainer.callback_handler.optimizer = trainer.callback_handler.lr_scheduler = None
    trainer._created_lr_scheduler = False
    trainer.control = TrainerControl()
    trainer.microbatch_records, trainer.logical_batch_sizes = [], []
    trainer.microbatch_peak_allocated = trainer.microbatch_peak_reserved = 0
    trainer._microbatch_loss_weight = 1.0
    for callback in trainer.callback_handler.callbacks:
        if callback.__class__.__name__ == "OptimizerStepTimer":
            callback.durations, callback.started_at = [], None
    restore_rng(rng)
    return tensor_digest(parameters)


def recover_candidate(
    trainer: Any, initial: list[torch.Tensor], rng: dict, expected_digest: str
) -> str:
    """Recover model/state after a scoped failure; reject a damaged CUDA context."""
    torch.cuda.synchronize()
    digest = reset_trainer(trainer, initial, rng)
    if digest != expected_digest:
        raise ValueError("failed candidate changed parameter layout")
    return digest


def prime_shapes(trainer: Any, loader: Any, expected: list[dict], path: Path) -> None:
    """Replay each trial batch once; actual baseline updates audit convergence."""
    records, sizes = trainer.microbatch_records, trainer.logical_batch_sizes
    prior = getattr(trainer, "current_gradient_accumulation_steps", None)
    steps = []
    try:
        trainer.microbatch_records, trainer.logical_batch_sizes = [], []
        trainer.current_gradient_accumulation_steps = 1
        with preserve_random_state():
            for step, inputs in enumerate(collect_batches(loader, 20), 1):
                trainer.model.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                start = time.perf_counter()
                loss = trainer.training_step(trainer.model_wrapped, inputs, None)
                torch.cuda.synchronize()
                if not math.isfinite(float(loss)):
                    raise FloatingPointError("nonfinite priming loss")
                steps.append({"step": step, "seconds": time.perf_counter() - start})
                write_json(path, {"status": "priming", "steps": steps})
                print(
                    f"resident_prime step={step}/20 seconds={steps[-1]['seconds']:.3f}",
                    flush=True,
                )
        contract = physical_contract(trainer.microbatch_records)
        if contract != expected:
            raise ValueError("resident priming physical contract changed")
        if trainer.optimizer.state:
            raise ValueError("priming initialized optimizer state")
        write_json(
            path, {"status": "complete", "steps": steps, "physical_contract": contract}
        )
    finally:
        trainer.model.zero_grad(set_to_none=True)
        trainer.microbatch_records, trainer.logical_batch_sizes = records, sizes
        if prior is None:
            del trainer.current_gradient_accumulation_steps
        else:
            trainer.current_gradient_accumulation_steps = prior


def resident_train(original_train: Callable, root: Path) -> Callable:
    """Suspend the normal entrypoint at Trainer.train while the worker is alive."""

    def train(trainer: Any, *args: Any, **kwargs: Any):
        if args or kwargs or trainer.args.max_steps != 20:
            raise ValueError("resident timing trials require twenty fresh updates")
        from torch._dynamo.utils import counters
        from transformers.trainer_utils import SaveStrategy
        from triton import knobs

        root.mkdir(parents=True, exist_ok=True)
        worker_source = Path(__file__).read_bytes()
        (root / "executed_resident_worker.py").write_bytes(worker_source)
        inbox = root / "requests"
        inbox.mkdir(exist_ok=True)
        reference = json.loads(
            Path(os.environ["GLEIPNIR_FP4_RESIDENT_REFERENCE"]).read_text()
        )
        reference_sha = (
            "14ab15279bb8895cf32353117d5c1cf957ad45d2b0ca9d7205067db27d77edeb"
        )
        expected = physical_contract(reference["adaptive_microbatching"]["records"])
        parameters = [p for p in trainer.model.parameters() if p.requires_grad]
        initial = [p.detach().cpu().clone() for p in parameters]
        write_json(
            root / "linear_modules.json",
            {
                "modules": [
                    {
                        "name": name,
                        "shape": list(module.weight.shape),
                        "dtype": str(module.weight.dtype),
                        "requires_grad": module.weight.requires_grad,
                    }
                    for name, module in trainer.model.named_modules()
                    if isinstance(module, torch.nn.Linear)
                ]
            },
        )
        initial_digest = tensor_digest(initial)
        if initial_digest != reference["sequence_packing"]["initial_master_sha256"]:
            raise ValueError("resident initial adapter drift")
        rng = capture_rng()
        trainer.args.save_strategy = SaveStrategy.NO
        trainer.args.disable_tqdm = True
        events: list[str] = []
        old_hook = knobs.runtime.jit_cache_hook

        def hook(**kw):
            events.append(kw["repr"])
            return old_hook(**kw) if old_hook else None

        def snapshot():
            return {
                "native_plans": cache_metadata()["plans"],
                "triton_specializations": len(events),
                "dynamo_unique_graphs": counters["stats"]["unique_graphs"],
                "inductor_fxgraph_misses": counters["inductor"]["fxgraph_cache_miss"],
            }

        primed = False
        audit: list[dict] = []
        current: dict = {}

        def status(state: str, **extra: Any):
            write_json(
                root / "worker.json",
                {
                    "status": state,
                    "pid": os.getpid(),
                    "worker_source_sha256": hashlib.sha256(worker_source).hexdigest(),
                    "trial": current.get("id"),
                    "native_cache": cache_metadata(),
                    **extra,
                },
            )

        class Audit(TrainerCallback):
            def on_train_begin(self, args, state, control, **kw):
                nonlocal primed
                if not primed:
                    status("priming")
                    prime_shapes(
                        trainer, kw["train_dataloader"], expected, root / "priming.json"
                    )
                    if tensor_digest(parameters) != initial_digest:
                        raise ValueError("priming changed FP32 masters")
                    primed = True
                status("training", update=0)

            def on_step_begin(self, args, state, control, **kw):
                self.before = snapshot()

            def on_step_end(self, args, state, control, **kw):
                delta = counter_delta(self.before, snapshot())
                audit.append({"step": state.global_step, "delta": delta})
                status("training", update=state.global_step, delta=delta)

        callback = Audit()
        trainer.add_callback(callback)
        knobs.runtime.jit_cache_hook = hook
        output = None
        try:
            status("ready")
            while True:
                queued = sorted(inbox.glob("*.json"))
                if not queued:
                    time.sleep(0.5)
                    continue
                request_path = queued[0]
                current = json.loads(request_path.read_text())
                validate_request(current)
                trial = root / current["id"]
                trial.mkdir(exist_ok=False)
                request_path.rename(trial / "request.json")
                try:
                    audit.clear()
                    started = time.perf_counter()
                    digest = reset_trainer(trainer, initial, rng)
                    if digest != initial_digest:
                        raise ValueError("resident master reset failed")
                    trainer.args.output_dir = str(trial / "adapter")
                    profile = None
                    if current.get("variant") == "gemmprofile":
                        from experiments.b200_mlp_gemm.resident_profile import (
                            GemmProfile,
                        )

                        profile = GemmProfile(trial)
                        trainer.add_callback(profile)
                    try:
                        intervention = (
                            candidate_context(trainer, current, trial)
                            if (current.get("variant") == "candidate")
                            else nullcontext()
                        )
                        with intervention:
                            output = original_train(trainer)
                    finally:
                        if profile is not None:
                            profile.close()
                            trainer.remove_callback(profile)
                    contract = physical_contract(trainer.microbatch_records)
                    if contract != expected or len(audit) != 20:
                        raise ValueError(
                            "resident trial physical contract/update count changed"
                        )
                    timing = next(
                        c
                        for c in trainer.callback_handler.callbacks
                        if c.__class__.__name__ == "OptimizerStepTimer"
                    )
                    samples = timing.durations
                    warm = all(
                        all(v == 0 for v in x["delta"].values()) for x in audit[10:]
                    )
                    final_digest = tensor_digest(parameters)
                    trainer.save_model(str(trial / "adapter"))
                    report = {
                        "status": "complete",
                        "variant": current.get("variant", "baseline"),
                        "pid": os.getpid(),
                        "initial_master_sha256": digest,
                        "final_master_sha256": final_digest,
                        "physical_contract": contract,
                        "loss_history": [
                            x for x in trainer.state.log_history if "loss" in x
                        ],
                        "step_seconds": samples,
                        "measured_mean_seconds": sum(samples[10:]) / 10,
                        "measured_updates_warm": warm,
                        "update_audit": audit.copy(),
                        "native_cache": cache_metadata(),
                        "wall_seconds": time.perf_counter() - started,
                        "instrumented": profile is not None
                        or (trial / "gemm_trace.json").exists(),
                        "optimizer_reset": True,
                        "startup_validation": {
                            "performed_this_run": False,
                            "reference_sha256": reference_sha,
                            "timing_only": True,
                            "original_strict_failures_preserved": True,
                        },
                    }
                    write_json(trial / "receipt.json", report)
                    if current["id"] == "02repeat":
                        baseline = json.loads(
                            (root / "01baseline/receipt.json").read_text()
                        )
                        keys = (
                            "initial_master_sha256",
                            "final_master_sha256",
                            "physical_contract",
                            "loss_history",
                        )
                        agreement = {key: report[key] == baseline[key] for key in keys}
                        write_json(root / "reset_validation.json", agreement)
                        if not all(agreement.values()):
                            raise ValueError(
                                "resident repeat did not reproduce the baseline"
                            )
                    if not warm:
                        raise ValueError(
                            "measured resident updates still prepare new shapes"
                        )
                    status("idle", last_receipt=str(trial / "receipt.json"))
                    print(
                        f"resident_trial_complete id={current['id']} "
                        f"mean={report['measured_mean_seconds']:.6f}",
                        flush=True,
                    )
                except Exception as error:
                    if current.get("variant") != "candidate":
                        raise
                    # A scoped candidate has restored its modules by this point.
                    # Failed CUDA contexts cannot safely remain resident.
                    restored = recover_candidate(trainer, initial, rng, initial_digest)
                    failure = {
                        "status": "failed",
                        "trial": current["id"],
                        "error": f"{type(error).__name__}: {error}",
                        "baseline_restored": True,
                        "initial_master_sha256": restored,
                    }
                    write_json(trial / "failure.json", failure)
                    status(
                        "idle", last_failed_trial=current["id"], error=failure["error"]
                    )
                    print(f"resident_candidate_failed {failure}", flush=True)
        except BaseException as error:
            status("failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            knobs.runtime.jit_cache_hook = old_hook
            trainer.remove_callback(callback)
        return output

    return train
