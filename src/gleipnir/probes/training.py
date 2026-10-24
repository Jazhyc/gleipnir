"""Train many sequence probes from one shared stream of activation batches.

Every probe sees the same minibatches in the same order; probe seeds change
only initialization. Reading and dequantizing a batch once serves all probes,
which keeps the I/O-bound activation store pass shared across the sweep.
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from .architectures import build
from .store import Store


@dataclass
class ProbeRun:
    name: str
    architecture: str
    objective: str
    seed: int
    probe: nn.Module
    optimizer: torch.optim.Optimizer
    losses: list[float] = field(default_factory=list)


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


def make_runs(
    architectures: Sequence[str],
    objectives: Sequence[str],
    seeds: Sequence[int],
    hidden: int,
    *,
    lr: float,
    weight_decay: float,
    device: str,
) -> list[ProbeRun]:
    """Matched initialization: one seed gives the same weights for every objective."""
    runs = []
    for architecture in architectures:
        for seed in seeds:
            for objective in objectives:
                torch.manual_seed(seed)
                probe = build(architecture, hidden).to(device)
                optimizer = torch.optim.AdamW(
                    probe.parameters(), lr=lr, weight_decay=weight_decay
                )
                name = f"{architecture}-{objective}-s{seed}"
                runs.append(
                    ProbeRun(name, architecture, objective, seed, probe, optimizer)
                )
    return runs


def targets(store: Store, objective: str) -> torch.Tensor:
    key = {"hard": "label", "soft": "soft_target"}[objective]
    return torch.tensor([float(r[key]) for r in store.rows])


def train_epoch(
    runs: list[ProbeRun], store: Store, batches: list[list[int]], device: str
) -> None:
    labels = {o: targets(store, o).to(device) for o in {r.objective for r in runs}}
    for indices, x, mask in stream(store, batches, device):
        for run in runs:
            run.probe.train()
            with torch.autocast(
                "cuda", torch.bfloat16, enabled=device.startswith("cuda")
            ):
                logits = run.probe(x, mask)
            loss = F.binary_cross_entropy_with_logits(
                logits.float(), labels[run.objective][indices]
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss in {run.name}")
            run.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            run.optimizer.step()
            run.losses.append(loss.item())


@torch.no_grad()
def score(
    runs: list[ProbeRun], store: Store, batch: int, device: str
) -> dict[str, list[float]]:
    """Return each probe's logit for every store row, in store order."""
    lengths = [r["length"] for r in store.rows]
    order = sorted(range(len(store)), key=lambda i: lengths[i])
    batches = [order[i : i + batch] for i in range(0, len(order), batch)]
    logits = {run.name: [0.0] * len(store) for run in runs}
    for indices, x, mask in stream(store, batches, device):
        for run in runs:
            run.probe.eval()
            with torch.autocast(
                "cuda", torch.bfloat16, enabled=device.startswith("cuda")
            ):
                values = run.probe(x, mask).float().tolist()
            for i, value in zip(indices, values, strict=True):
                logits[run.name][i] = value
    return logits
