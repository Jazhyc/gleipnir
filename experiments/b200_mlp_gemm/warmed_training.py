"""Same-process shape warmup without optimizer updates or sampler/RNG drift."""

from __future__ import annotations

import json
import random
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from gleipnir.cudnn_fp4_mlp import cache_metadata
from gleipnir.training_execution_audit import tensor_digest


def physical_contract(records: list[dict]) -> list[dict]:
    return [
        {k: r[k] for k in ("update", "logical_indices", "tokens", "padded_tokens")}
        for r in records
    ]


def counter_delta(before: dict, after: dict) -> dict:
    return {k: after[k] - before[k] for k in before}


def is_warm_delta(delta: dict) -> bool:
    """No new native plans, Triton specializations or decoder compiler graphs."""
    return bool(delta) and all(v == 0 for v in delta.values())


def validate_warmed_receipt(receipt: dict, candidate: dict) -> None:
    """Never call a run warmed if measured updates still specialize or plan."""
    if (
        receipt.get("status") != "warmup_complete"
        or receipt.get("masters_unchanged") is not True
        or receipt.get("optimizer_state_unchanged") is not True
        or receipt.get("optimizer_updates") != 0
        or receipt.get("initial_master_sha256") != candidate["initial_master_sha256"]
        or receipt.get("final_master_sha256") != candidate["initial_master_sha256"]
        or len(receipt.get("passes", [])) < 2
        or receipt["passes"][-1].get("physical_contract")
        != candidate["physical_contract"]
        or not is_warm_delta(receipt["passes"][-1].get("delta", {}))
    ):
        raise ValueError("warmup did not establish unchanged-master shape coverage")
    updates = receipt.get("update_audit", [])
    if [r["step"] for r in updates] != list(range(1, 21)) or any(
        not is_warm_delta(r["delta"]) for r in updates[10:]
    ):
        raise ValueError("measured updates still compile or construct native plans")


@contextmanager
def preserve_random_state():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = (
        list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    )
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def collect_batches(loader, steps: int) -> list[dict]:
    """Replay the Trainer's epoch-seeded sampler; restore it before training."""
    from accelerate.data_loader import SeedableRandomSampler

    if loader.num_workers != 0 or not isinstance(
        loader.get_sampler(), SeedableRandomSampler
    ):
        raise ValueError("shape warmup requires the validated epoch-seeded loader")
    sampler = loader.get_sampler()
    initial_iteration, initial_epoch = loader.iteration, sampler.epoch
    generator = sampler.generator
    generator_state = generator.get_state() if generator is not None else None
    batches = []
    try:
        with preserve_random_state():
            for epoch in range((steps + len(loader) - 1) // len(loader)):
                loader.set_epoch(epoch)
                for inputs in loader:
                    batches.append(inputs)
                if len(batches) >= steps:
                    break
    finally:
        loader.set_epoch(initial_iteration)
        sampler.set_epoch(initial_epoch)
        sampler.generator = generator
        if generator is not None:
            generator.set_state(generator_state)
    if len(batches) < steps:
        raise ValueError("incomplete warmup batch sequence")
    return batches[:steps]


def warm_shapes(trainer, batches, snapshot, expected_contract, path: Path) -> dict:
    """Warm full forward/backward until replay creates no new compiled shapes."""
    if len(batches) != 20 or trainer.state.global_step != 0:
        raise ValueError("shape warmup requires a fresh twenty-update trajectory")
    model = trainer.model_wrapped
    parameters = [p for p in model.parameters() if p.requires_grad]
    initial = tensor_digest(parameters)
    saved = {
        name: deepcopy(getattr(trainer, name))
        for name in (
            "microbatch_records",
            "logical_batch_sizes",
            "microbatch_peak_allocated",
            "microbatch_peak_reserved",
            "_microbatch_loss_weight",
        )
    }
    prior_accumulation = getattr(trainer, "current_gradient_accumulation_steps", None)
    optimizer_state = len(trainer.optimizer.state)
    if optimizer_state:
        raise ValueError("warmup must precede optimizer state initialization")
    report = {
        "status": "warming",
        "initial_master_sha256": initial,
        "optimizer_updates": 0,
        "passes": [],
    }

    def save():
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def synchronize():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    save()
    try:
        with preserve_random_state():
            trainer.current_gradient_accumulation_steps = 1
            for repetition in range(3):
                trainer.microbatch_records, trainer.logical_batch_sizes = [], []
                before = snapshot()
                current = {"pass": repetition + 1, "steps": [], "before": before}
                report["passes"].append(current)
                for step, inputs in enumerate(batches, 1):
                    model.zero_grad(set_to_none=True)
                    synchronize()
                    started = time.perf_counter()
                    loss = trainer.training_step(model, inputs, None)
                    synchronize()
                    if not torch.isfinite(loss).item():
                        raise FloatingPointError("nonfinite warmup loss")
                    gradients = [p.grad for p in parameters]
                    if not gradients or any(g is None for g in gradients):
                        raise FloatingPointError("missing warmup adapter gradient")
                    torch.nn.utils.get_total_norm(gradients, error_if_nonfinite=True)
                    current["steps"].append(
                        {
                            "step": step,
                            "seconds": time.perf_counter() - started,
                            "loss": float(loss),
                            "counters": snapshot(),
                        }
                    )
                    save()
                    print(
                        f"fp4_shape_warmup pass={repetition + 1} step={step}/20 "
                        f"seconds={current['steps'][-1]['seconds']:.3f}",
                        flush=True,
                    )
                current["physical_contract"] = physical_contract(
                    trainer.microbatch_records
                )
                if current["physical_contract"] != expected_contract:
                    raise ValueError("warmup physical batches differ from FA4 control")
                current["after"] = snapshot()
                current["delta"] = counter_delta(before, current["after"])
                current["warm"] = is_warm_delta(current["delta"])
                save()
                if repetition >= 1 and current["warm"]:
                    break
            else:
                raise ValueError("shape replay still compiles after three passes")
        final = tensor_digest(parameters)
        if final != initial or len(trainer.optimizer.state) != optimizer_state:
            raise ValueError("warmup changed adapters or optimizer state")
        report.update(
            status="warmup_complete",
            final_master_sha256=final,
            masters_unchanged=True,
            optimizer_state_unchanged=True,
        )
        save()
        return report
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise
    finally:
        model.zero_grad(set_to_none=True)
        for name, value in saved.items():
            setattr(trainer, name, value)
        if prior_accumulation is None:
            del trainer.current_gradient_accumulation_steps
        else:
            trainer.current_gradient_accumulation_steps = prior_accumulation


def install_warmed_train(original_train, reference: Path, report_path: Path):
    """Attach warmup after Trainer preparation, plus per-update compile auditing."""
    from transformers import TrainerCallback
    from triton import knobs

    expected = physical_contract(
        json.loads(reference.read_text())["adaptive_microbatching"]["records"]
    )

    def train(trainer, *args, **kwargs):
        if args or kwargs or trainer.args.max_steps != 20:
            raise ValueError("warmed screen requires an unchanged fresh 20-step train")
        events, records = [], []
        previous_hook = knobs.runtime.jit_cache_hook

        def hook(**kw):
            events.append({"kernel": kw["repr"], "time": time.time()})
            return previous_hook(**kw) if previous_hook else None

        def snapshot():
            from torch._dynamo.utils import counters

            return {
                "native_plans": cache_metadata()["plans"],
                "triton_specializations": len(events),
                "dynamo_unique_graphs": counters["stats"]["unique_graphs"],
                "inductor_fxgraph_misses": counters["inductor"]["fxgraph_cache_miss"],
            }

        class WarmupAudit(TrainerCallback):
            def on_train_begin(self, args, state, control, **kw):
                warm_shapes(
                    trainer,
                    collect_batches(kw["train_dataloader"], 20),
                    snapshot,
                    expected,
                    report_path,
                )

            def on_step_begin(self, args, state, control, **kw):
                self.before = snapshot()

            def on_step_end(self, args, state, control, **kw):
                after = snapshot()
                delta = counter_delta(self.before, after)
                records.append(
                    {
                        "step": state.global_step,
                        "before": self.before,
                        "after": after,
                        "delta": delta,
                        "warm": is_warm_delta(delta),
                    }
                )
                report = json.loads(report_path.read_text())
                report.update(update_audit=records, triton_specialization_events=events)
                report_path.write_text(json.dumps(report, indent=2) + "\n")

        callback = WarmupAudit()
        trainer.add_callback(callback)
        try:
            knobs.runtime.jit_cache_hook = hook
            return original_train(trainer)
        finally:
            knobs.runtime.jit_cache_hook = previous_hook
            trainer.remove_callback(callback)

    return train
