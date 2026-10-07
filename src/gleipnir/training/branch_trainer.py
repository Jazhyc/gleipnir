"""Parent-balanced, resumable model-parallel all-prefix training."""

import hashlib
import json
import math
import time
from pathlib import Path

import torch

from gleipnir.branch_training import branched_decision_logits, plan_branches
from gleipnir.monitoring_systems_screen import atomic_write_json
from gleipnir.prefix_loss import trajectory_prefix_loss


def parent_objective(logits: torch.Tensor, targets: list[float], weight: float):
    """Mean individual prefix BCEs, then combine with the original full BCE."""
    target = torch.tensor(targets, device=logits.device, dtype=torch.float32)
    if not torch.isfinite(target).all() or ((target < 0) | (target > 1)).any():
        raise ValueError("invalid soft target")
    losses = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[:, 1].float() - logits[:, 0].float(), target, reduction="none"
    )
    objective = trajectory_prefix_loss(
        losses[-1:],
        losses[:-1],
        torch.zeros(len(targets) - 1, device=logits.device, dtype=torch.long),
        weight,
    )
    return objective, losses[-1], losses[:-1].mean() if len(targets) > 1 else None


def accumulation_windows(order: list[int], batch_size: int):
    """The final short window has its own denominator, not the nominal batch."""
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    for offset in range(0, len(order), batch_size):
        yield offset, order[offset : offset + batch_size]


def synchronize() -> None:
    for device in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device)


def budget_lr_multiplier(fraction: float, warmup_ratio: float) -> float:
    """Linear warmup/decay evaluated at the midpoint of an update's work."""
    if not math.isfinite(fraction) or not 0 <= warmup_ratio < 1:
        raise ValueError("invalid time-based schedule")
    fraction = min(max(fraction, 0.0), 1.0)
    if fraction < warmup_ratio:
        return fraction / warmup_ratio
    return (1.0 - fraction) / (1.0 - warmup_ratio)


def normalize_partial_window(params, nominal: int, completed: int) -> None:
    """Undo the nominal accumulation denominator at a budget-limited endpoint."""
    if not 0 < completed <= nominal:
        raise ValueError("invalid partial optimizer window")
    if completed != nominal:
        for parameter in params:
            if parameter.grad is not None:
                parameter.grad.mul_(nominal / completed)


def save_checkpoint(model, optimizer, scheduler, root: Path, state: dict) -> None:
    """Publish only complete checkpoints; preserve all earlier recovery points."""
    directory = root / "checkpoints" / f"step-{state['step']:04d}"
    directory.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(directory / "adapter", safe_serialization=True)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            **state,
        },
        directory / "state.pt",
    )
    atomic_write_json(directory / "complete.json", state)
    atomic_write_json(
        root / "latest_checkpoint.json", {"path": str(directory), **state}
    )


def train(
    model,
    dataset,
    config: dict,
    root: Path,
    *,
    resume: Path | None = None,
    smoke_parents: int | None = None,
) -> dict:
    """Run one frozen parent epoch with no prefix sampling or graph detachment."""
    from transformers import get_linear_schedule_with_warmup

    budget = None if smoke_parents else config.get("training_wall_seconds")
    if budget is not None:
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError("training budget must be finite and positive")
        if resume is not None:
            raise ValueError("budgeted resume requires accounting for discarded work")
    params = [p for p in model.parameters() if p.requires_grad]
    contract_sha = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()
    order = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(config["seed"])
    ).tolist()
    if smoke_parents is not None:
        order = order[:smoke_parents]
    total_steps = math.ceil(len(order) / config["effective_batch"])
    optimizer = torch.optim.AdamW(
        params,
        lr=config["learning_rate"],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        fused=False,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        0 if smoke_parents else math.ceil(total_steps * config["warmup_ratio"]),
        total_steps,
    )
    state = {"step": 0, "parents": 0, "prefixes": 0, "contract_sha256": contract_sha}
    if resume is not None:
        saved = torch.load(resume / "state.pt", map_location="cpu", weights_only=True)
        if saved["contract_sha256"] != contract_sha:
            raise ValueError("resume contract drift")
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        state.update({k: saved[k] for k in state})
        if state["parents"] not in [
            o for o, _ in accumulation_windows(order, config["effective_batch"])
        ] + [len(order)]:
            raise ValueError("resume outside optimizer boundary")
    atomic_write_json(root / "order.json", order)
    started = time.perf_counter()
    starting_parents = state["parents"]
    budget_reached = False
    for offset, window in accumulation_windows(order, config["effective_batch"]):
        if offset < state["parents"]:
            continue
        if budget is not None and time.perf_counter() - started >= budget:
            budget_reached = True
            break
        optimizer.zero_grad(set_to_none=True)
        step_started = time.perf_counter()
        total_loss, full_loss, prefix_loss, eligible = 0.0, 0.0, 0.0, 0
        completed = 0
        for parent_index in window:
            if budget is not None and time.perf_counter() - started >= budget:
                budget_reached = True
                break
            parent_started = time.perf_counter()
            row = dataset[parent_index]
            plan = plan_branches(row["plan"].requests, alignment=config["alignment"])
            print(
                json.dumps(
                    {
                        "event": "parent_start",
                        "parent": row["parent_id"],
                        "prefixes": len(row["targets"]) - 1,
                    }
                ),
                flush=True,
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = branched_decision_logits(
                    model,
                    plan,
                    [15, 16],
                    checkpoint_segments=True,
                    fp32_head=True,
                    independent_endpoint=True,
                )
                objective, full, prefix = parent_objective(
                    logits, row["targets"], config["prefix_weight"]
                )
            if not torch.isfinite(objective):
                raise RuntimeError(f"nonfinite parent loss: {row['parent_id']}")
            (objective / len(window)).backward()
            total_loss += float(objective.detach())
            full_loss += float(full.detach())
            if prefix is not None:
                eligible += 1
                prefix_loss += float(prefix.detach())
            state["prefixes"] += len(row["targets"]) - 1
            # Drop every reference to the graph before the next parent forward.
            del logits, objective, full, prefix
            completed += 1
            synchronize()
            print(
                json.dumps(
                    {
                        "event": "parent_complete",
                        "parent": row["parent_id"],
                        "seconds": time.perf_counter() - parent_started,
                    }
                ),
                flush=True,
            )
        if not completed:
            break
        normalize_partial_window(params, len(window), completed)
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):
            raise RuntimeError("missing or nonfinite accumulated gradient")
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        if budget is not None:
            # A short-budget run must not inherit the 272-update epoch schedule.
            midpoint = ((step_started - started) + (time.perf_counter() - started)) / 2
            multiplier = budget_lr_multiplier(midpoint / budget, config["warmup_ratio"])
            for group in optimizer.param_groups:
                group["lr"] = config["learning_rate"] * multiplier
        lr = optimizer.param_groups[0]["lr"]
        optimizer.step()
        if budget is None:
            scheduler.step()
        if any(not torch.isfinite(p).all() for p in params):
            raise RuntimeError("nonfinite master after optimizer update")
        synchronize()
        state.update(step=state["step"] + 1, parents=offset + completed)
        seconds = time.perf_counter() - started
        if budget is not None and seconds >= budget:
            budget_reached = True
        record = {
            **state,
            "event": "optimizer_step",
            "total_steps": total_steps if budget is None else None,
            "loss": total_loss / completed,
            "full_loss": full_loss / completed,
            "prefix_loss_eligible_mean": prefix_loss / eligible if eligible else None,
            "gradient_norm": float(norm),
            "learning_rate": lr,
            "step_seconds": time.perf_counter() - step_started,
            "seconds": seconds,
            "estimated_remaining_seconds": (
                max(0.0, budget - seconds)
                if budget is not None
                else seconds
                / (state["parents"] - starting_parents)
                * (len(order) - state["parents"])
            ),
            "training_gpu_seconds": seconds * 2,
            "target_training_wall_seconds": budget,
            "peak_allocated_bytes": [
                torch.cuda.max_memory_allocated(i) for i in range(2)
            ],
        }
        with (root / "progress.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        atomic_write_json(root / "status.json", {"state": "training", **record})
        print(json.dumps(record), flush=True)
        if not smoke_parents and (
            state["step"] % config["checkpoint_every"] == 0
            or state["parents"] == len(order)
            or budget_reached
        ):
            save_checkpoint(model, optimizer, scheduler, root, state)
        if budget_reached:
            break
    if (
        not smoke_parents
        and budget is None
        and (state["parents"] != 8688 or state["prefixes"] != 133947)
    ):
        raise RuntimeError("incomplete parent/prefix coverage")
    if smoke_parents and not any(
        bool(p.detach().ne(0).any())
        for n, p in model.named_parameters()
        if "lora_B" in n
    ):
        raise RuntimeError("fresh-adapter smoke update had no adapter effect")
    return {
        **state,
        "seconds_this_session": time.perf_counter() - started,
        "smoke_only": smoke_parents is not None,
        "budget_reached": budget_reached,
        "target_training_wall_seconds": budget,
        "training_gpu_seconds": (time.perf_counter() - started) * 2,
        "budget_overrun_seconds": (
            max(0.0, time.perf_counter() - started - budget) if budget else 0.0
        ),
    }
