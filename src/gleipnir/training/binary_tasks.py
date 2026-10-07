"""Explicit per-example binary tasks and reproducible monitoring-anchored sampling."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Sampler

from gleipnir.decision_surface import decision_token_ids


def binary_task_feature(record: dict, tokenizer: Any) -> dict:
    """Keep hard preference labels separate from privileged monitoring targets."""
    group = record["sampling_group"]
    if group == "monitor":
        tokens, prefix, objective = ["0", "1"], "Prediction:", "soft"
    elif group in {"clean", "preferred_injected", "disfavored_injected"}:
        tokens, prefix, objective = ["A", "B"], "", "hard"
    else:
        raise ValueError(f"unknown binary sampling group: {group}")
    if (
        record["decision_tokens"] != tokens
        or record["decision_prefix"] != prefix
        or record["binary_objective"] != objective
        or record["label"] not in (0, 1)
    ):
        raise ValueError("binary task surface/objective drift")
    if objective == "soft":
        target = float(record["_soft_target"])
        if not math.isfinite(target) or not 0 <= target <= 1:
            raise ValueError("invalid privileged soft target")
    else:
        if "_soft_target" in record:
            raise ValueError("hard preference row cannot contain a teacher target")
        target = 0.0  # Explicitly masked; never a preference training target.
    return {
        "row_decision_token_ids": decision_token_ids(tokenizer, tokens),
        "hard_objective": objective == "hard",
        "soft_target": target,
        "sampling_group": group,
    }


def collate_binary_task_fields(features: Sequence[dict]) -> dict:
    """Preserve task readouts when a packed partition reorders examples."""
    present = ["row_decision_token_ids" in f for f in features]
    if not any(present):
        return {}
    if not all(present):
        raise ValueError("mixed missing per-record binary surfaces")
    return {
        "row_decision_token_ids": torch.tensor(
            [f["row_decision_token_ids"] for f in features], dtype=torch.long
        ),
        "hard_objectives": torch.tensor(
            [f["hard_objective"] for f in features], dtype=torch.bool
        ),
    }


def binary_task_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    soft_targets: torch.Tensor,
    hard_objectives: torch.Tensor,
) -> torch.Tensor:
    """One example, one loss: hard CE or the original soft binary BCE."""
    if logits.shape != (len(labels), 2) or any(
        value.shape != labels.shape for value in (soft_targets, hard_objectives)
    ):
        raise ValueError("binary task loss shape drift")
    hard = F.cross_entropy(logits.float(), labels, reduction="none")
    soft = F.binary_cross_entropy_with_logits(
        logits[:, 1].float() - logits[:, 0].float(),
        soft_targets.float(),
        reduction="none",
    )
    return torch.where(hard_objectives, hard, soft).mean()


class TaskMixtureSampler(Sampler[int]):
    """Visit every monitor once; draw a fixed auxiliary quota each logical batch."""

    def __init__(
        self,
        groups: Sequence[str],
        labels: Sequence[int],
        *,
        seed: int,
        monitoring_per_batch: int = 24,
        preference_per_batch: int = 8,
        condition_weights: dict[str, float] | None = None,
    ) -> None:
        self.groups, self.labels = list(groups), list(labels)
        self.seed, self.epoch = seed, 0
        self.monitoring_per_batch = monitoring_per_batch
        self.preference_per_batch = preference_per_batch
        self.weights = condition_weights or {
            "clean": 0.10,
            "preferred_injected": 0.45,
            "disfavored_injected": 0.45,
        }
        if (
            len(groups) != len(labels)
            or not groups
            or monitoring_per_batch < 1
            or preference_per_batch < 1
            or set(self.weights)
            != {"clean", "preferred_injected", "disfavored_injected"}
            or any(not math.isfinite(w) or w <= 0 for w in self.weights.values())
            or not math.isclose(sum(self.weights.values()), 1.0)
            or set(groups) != {"monitor", *self.weights}
            or any(label not in (0, 1) for label in labels)
        ):
            raise ValueError("invalid task mixture contract")
        self.monitors = [i for i, group in enumerate(groups) if group == "monitor"]
        self.pools = defaultdict(list)
        for i, (group, label) in enumerate(zip(groups, labels, strict=True)):
            if group != "monitor":
                self.pools[group, label].append(i)
        if any(not self.pools[g, label] for g in self.weights for label in (0, 1)):
            raise ValueError("every preference condition needs both A/B orders")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return (
            len(self.monitors)
            + math.ceil(len(self.monitors) / self.monitoring_per_batch)
            * self.preference_per_batch
        )

    def indices(self) -> list[int]:
        rng = random.Random(self.seed + self.epoch)
        monitors = self.monitors.copy()
        rng.shuffle(monitors)
        auxiliary_count = len(self) - len(monitors)
        quotas = {g: math.floor(auxiliary_count * w) for g, w in self.weights.items()}
        remaining = auxiliary_count - sum(quotas.values())
        order = sorted(
            self.weights,
            key=lambda g: (-(auxiliary_count * self.weights[g] - quotas[g]), g),
        )
        for g in order[:remaining]:
            quotas[g] += 1
        conditions = [g for g, count in quotas.items() for _ in range(count)]
        rng.shuffle(conditions)
        pools = {key: values.copy() for key, values in self.pools.items()}
        for pool in pools.values():
            rng.shuffle(pool)
        cursors: Counter = Counter()
        next_labels = {g: rng.randrange(2) for g in self.weights}
        auxiliaries = []
        for group in conditions:
            label = next_labels[group]
            next_labels[group] = 1 - label
            key = (group, label)
            if cursors[key] == len(pools[key]):
                rng.shuffle(pools[key])
                cursors[key] = 0
            auxiliaries.append(pools[key][cursors[key]])
            cursors[key] += 1
        result = []
        for number, start in enumerate(
            range(0, len(monitors), self.monitoring_per_batch)
        ):
            batch = (
                monitors[start : start + self.monitoring_per_batch]
                + auxiliaries[
                    number * self.preference_per_batch : (number + 1)
                    * self.preference_per_batch
                ]
            )
            rng.shuffle(batch)
            result.extend(batch)
        return result

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices())

    def audit(self) -> dict:
        indices = self.indices()
        counts = Counter(self.groups[i] for i in indices)
        return {
            "seed": self.seed,
            "epoch": self.epoch,
            "samples": len(indices),
            "group_counts": dict(counts),
            "unique_monitor_rows": len(
                {i for i in indices if self.groups[i] == "monitor"}
            ),
            "unique_preference_rows": len(
                {i for i in indices if self.groups[i] != "monitor"}
            ),
            "order_sha256": hashlib.sha256(json.dumps(indices).encode()).hexdigest(),
            "loss_normalization": "equal_examples_with_task_specific_binary_targets",
        }
