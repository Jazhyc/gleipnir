"""Train a sweep of sequence probes from one shared stream of activation batches.

Every probe sees the same minibatches in the same order; probe seeds change
only initialization. The whole sweep runs as one fused :class:`ProbeBank`, so
reading a batch once serves all probes and their first layers share one matmul.
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .architectures import build
from .bank import ProbeBank
from .store import Store


@dataclass
class ProbeSpec:
    name: str
    architecture: str
    objective: str
    seed: int
    losses: list[float] = field(default_factory=list)


@dataclass
class Sweep:
    specs: list[ProbeSpec]
    bank: ProbeBank
    optimizer: torch.optim.Optimizer


def length_batches(
    lengths: Sequence[int], batch: int, seed: int, chunk: int = 256
) -> list[list[int]]:
    """Shuffle, sort within chunks by length to limit padding, shuffle batches."""
    rng = random.Random(seed)
    order = list(range(len(lengths)))
    rng.shuffle(order)
    batches = []
    for begin in range(0, len(order), chunk):
        part = sorted(order[begin : begin + chunk], key=lambda i: lengths[i])
        batches.extend(part[i : i + batch] for i in range(0, len(part), batch))
    rng.shuffle(batches)
    return batches


def pad(rows: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    longest = max(len(r) for r in rows)
    x = rows[0].new_zeros((len(rows), longest, rows[0].shape[1]))
    mask = torch.zeros((len(rows), longest), dtype=torch.bool, device=rows[0].device)
    for i, r in enumerate(rows):
        x[i, : len(r)] = r
        mask[i, : len(r)] = True
    return x, mask


def stream(
    store: Store, batches: list[list[int]], device: str
) -> Iterator[tuple[list[int], torch.Tensor, torch.Tensor]]:
    """Yield padded BF16 batches; on CPU (tests), upcast since autocast is off."""
    for indices in batches:
        x, mask = pad([store.load(i, device) for i in indices])
        yield indices, (x if device.startswith("cuda") else x.float()), mask


def make_sweep(
    architectures: Sequence[str],
    objectives: Sequence[str],
    seeds: Sequence[int],
    hidden: int,
    *,
    lr: float,
    weight_decay: float,
    device: str,
) -> Sweep:
    """Matched initialization: one seed gives the same weights for every objective."""
    specs, probes = [], []
    for architecture in architectures:
        for seed in seeds:
            for objective in objectives:
                torch.manual_seed(seed)
                probes.append(build(architecture, hidden))
                name = f"{architecture}-{objective}-s{seed}"
                specs.append(ProbeSpec(name, architecture, objective, seed))
    bank = ProbeBank([s.architecture for s in specs], probes).to(device)
    optimizer = torch.optim.AdamW(bank.parameters(), lr=lr, weight_decay=weight_decay)
    return Sweep(specs, bank, optimizer)


def targets(store: Store, specs: Sequence[ProbeSpec]) -> torch.Tensor:
    """``[rows, probes]`` training targets: source labels or Kimi soft targets."""
    keys = [{"hard": "label", "soft": "soft_target"}[s.objective] for s in specs]
    return torch.tensor([[float(r[k]) for k in keys] for r in store.rows])


def autocast(device: str):
    return torch.autocast("cuda", torch.bfloat16, enabled=device.startswith("cuda"))


def train_epoch(
    sweep: Sweep, store: Store, batches: list[list[int]], device: str
) -> None:
    labels = targets(store, sweep.specs).to(device)
    sweep.bank.train()
    for indices, x, mask in stream(store, batches, device):
        with autocast(device):
            logits = sweep.bank(x, mask)
        # Mean over rows per probe, summed over probes: parameters are disjoint,
        # so each probe's gradient equals its own single-probe gradient.
        losses = F.binary_cross_entropy_with_logits(
            logits.float(), labels[indices], reduction="none"
        ).mean(0)
        if not bool(torch.isfinite(losses).all()):
            bad = [
                s.name
                for s, v in zip(sweep.specs, losses, strict=True)
                if not torch.isfinite(v)
            ]
            raise FloatingPointError(f"nonfinite loss in {bad}")
        sweep.optimizer.zero_grad(set_to_none=True)
        losses.sum().backward()
        sweep.optimizer.step()
        for spec, value in zip(sweep.specs, losses.tolist(), strict=True):
            spec.losses.append(value)


@torch.no_grad()
def score(
    sweep: Sweep, store: Store, batch: int, device: str
) -> dict[str, list[float]]:
    """Return each probe's logit for every store row, in store order."""
    lengths = [r["length"] for r in store.rows]
    order = sorted(range(len(store)), key=lambda i: lengths[i])
    batches = [order[i : i + batch] for i in range(0, len(order), batch)]
    logits = torch.zeros((len(store), len(sweep.specs)))
    sweep.bank.eval()
    for indices, x, mask in stream(store, batches, device):
        with autocast(device):
            logits[indices] = sweep.bank(x, mask).float().cpu()
    return {s.name: logits[:, i].tolist() for i, s in enumerate(sweep.specs)}
