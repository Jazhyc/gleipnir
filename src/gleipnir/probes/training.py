"""Train a sweep of sequence probes from one shared stream of activation batches.

Every probe sees the same minibatches in the same order; probe seeds change
only initialization. The whole sweep runs as one fused :class:`ProbeBank`, so
reading a batch once serves all probes and their first layers share one matmul.
"""

from __future__ import annotations

import random
from collections import deque
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
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


PREFETCH_BATCHES = 4
PREFETCH_WORKERS = 8


def stream(
    store: Store, batches: list[list[int]], device: str
) -> Iterator[tuple[list[int], torch.Tensor, torch.Tensor]]:
    """Yield padded BF16 batches; on CPU (tests), upcast since autocast is off.

    Disk-backed stores read upcoming batches' raw rows on background threads
    while the current batch computes; transfer and dequantization stay on the
    caller's device.
    """
    upcast = not device.startswith("cuda")
    if not isinstance(store, Store):
        for indices in batches:
            x, mask = pad([store.load(i, device) for i in indices])
            yield indices, (x.float() if upcast else x), mask
        return
    with ThreadPoolExecutor(PREFETCH_WORKERS) as pool:
        pending: deque = deque()
        upcoming = iter(batches)

        def submit() -> None:
            indices = next(upcoming, None)
            if indices is not None:
                pending.append(
                    (indices, [pool.submit(store.read_raw, i) for i in indices])
                )

        for _ in range(PREFETCH_BATCHES):
            submit()
        while pending:
            indices, futures = pending.popleft()
            submit()
            rows = [store.from_raw(f.result(), device) for f in futures]
            x, mask = pad(rows)
            yield indices, (x.float() if upcast else x), mask


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


def micro_batches(
    indices: list[int], lengths: Sequence[int], token_budget: int | None
) -> list[list[int]]:
    """Split a batch into consecutive chunks whose padded size fits the budget."""
    if token_budget is None:
        return [indices]
    chunks, current = [], []
    for i in indices:
        longest = max([lengths[j] for j in current] + [lengths[i]])
        if current and longest * (len(current) + 1) > token_budget:
            chunks.append(current)
            current = []
        current.append(i)
    return chunks + [current]


def train_epoch(
    sweep: Sweep,
    store: Store,
    batches: list[list[int]],
    device: str,
    token_budget: int | None = None,
) -> None:
    """One pass; ``token_budget`` caps padded tokens per forward (gradient
    accumulation within a batch, same update as the unsplit batch)."""
    labels = targets(store, sweep.specs).to(device)
    lengths = [r["length"] for r in store.rows]
    sweep.bank.train()
    for batch in batches:
        sweep.optimizer.zero_grad(set_to_none=True)
        losses = torch.zeros(len(sweep.specs), device=device)
        for indices, x, mask in stream(
            store, micro_batches(batch, lengths, token_budget), device
        ):
            with autocast(device):
                logits = sweep.bank(x, mask)
            # Mean over the batch's rows per probe, summed over probes:
            # parameters are disjoint, so each probe's gradient equals its
            # single-probe gradient. Logged losses are BCE only.
            part = F.binary_cross_entropy_with_logits(
                logits.float(), labels[indices], reduction="none"
            ).sum(0) / len(batch)
            part.sum().backward()
            losses += part.detach()
        if not bool(torch.isfinite(losses).all()):
            bad = [
                s.name
                for s, v in zip(sweep.specs, losses, strict=True)
                if not torch.isfinite(v)
            ]
            raise FloatingPointError(f"nonfinite loss in {bad}")
        penalty = sweep.bank.regularization().sum()
        if penalty.requires_grad:
            penalty.backward()
        sweep.optimizer.step()
        for spec, value in zip(sweep.specs, losses.tolist(), strict=True):
            spec.losses.append(value)


@torch.no_grad()
def score(
    sweep: Sweep,
    store: Store,
    batch: int,
    device: str,
    token_budget: int | None = None,
) -> dict[str, list[float]]:
    """Return each probe's logit for every store row, in store order."""
    lengths = [r["length"] for r in store.rows]
    order = sorted(range(len(store)), key=lambda i: lengths[i])
    batches = [
        chunk
        for begin in range(0, len(order), batch)
        for chunk in micro_batches(order[begin : begin + batch], lengths, token_budget)
    ]
    logits = torch.zeros((len(store), len(sweep.specs)))
    sweep.bank.eval()
    for indices, x, mask in stream(store, batches, device):
        with autocast(device):
            logits[indices] = sweep.bank(x, mask).float().cpu()
    return {s.name: logits[:, i].tolist() for i, s in enumerate(sweep.specs)}
