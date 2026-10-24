"""Many sequence probes evaluated as one fused module.

Every probe in a sweep reads the same activations, and nearly all compute is
each probe's first token layer (``hidden -> 100``). The bank stacks those
layers, and the ``mean_linear`` readouts, into one ``hidden -> sum(width)``
matmul; second layers run as one batched matmul and each architecture's heads
are pooled together by the shared functions in :mod:`.architectures`.

Parameters stay disjoint per probe, so summing per-probe losses and stepping
one elementwise optimizer (AdamW) trains each probe exactly as if it were
alone. Probes are initialized from their single-probe modules
(:func:`.architectures.build`), so a seed gives the same starting weights.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .architectures import (
    HEADS,
    WIDTH,
    HeadProbe,
    MeanProbe,
    attention_pool,
    masked_mean,
    max_pool,
    rolling_pool,
)


def _project(
    members: torch.Tensor, weight: torch.Tensor, batch: int, tokens: int
) -> torch.Tensor:
    """``[B*T, P, WIDTH]`` x ``[P, heads, WIDTH]`` -> ``[B, P*heads, T]``."""
    heads = torch.einsum("npw,phw->nph", members, weight).float()
    return heads.reshape(batch, tokens, -1).transpose(1, 2)


def _stack(tensors: list[torch.Tensor]) -> nn.Parameter:
    return nn.Parameter(torch.stack([t.detach().clone() for t in tensors]))


class ProbeBank(nn.Module):
    def __init__(self, architectures: Sequence[str], probes: Sequence[nn.Module]):
        super().__init__()
        if len(architectures) != len(probes):
            raise ValueError("one architecture name per probe")
        self.architectures = list(architectures)
        linear = [i for i, a in enumerate(architectures) if a == "mean_linear"]
        deep = [i for i, a in enumerate(architectures) if a != "mean_linear"]
        self.linear_index, self.deep_index = linear, deep
        first_w, first_b = [], []
        for i in linear:
            first_w.append(probes[i].score.weight)  # [1, hidden]
            first_b.append(probes[i].score.bias)
        for i in deep:
            first_w.append(probes[i].body[0].weight)  # [WIDTH, hidden]
            first_b.append(probes[i].body[0].bias)
        self.first_weight = nn.Parameter(torch.cat(first_w).detach().clone())
        self.first_bias = nn.Parameter(torch.cat(first_b).detach().clone())
        if deep:
            self.second_weight = _stack([probes[i].body[2].weight.T for i in deep])
            self.second_bias = _stack([probes[i].body[2].bias for i in deep])
        # Heads, grouped by architecture over positions in ``deep``.
        self.groups: dict[str, list[int]] = {}
        for position, i in enumerate(deep):
            self.groups.setdefault(architectures[i], []).append(position)
        self.heads_params = nn.ParameterDict()
        for name, positions in self.groups.items():
            members = [probes[deep[p]] for p in positions]
            key = name.replace("-", "_")
            if name == "mean_mlp":
                if not all(isinstance(m, MeanProbe) for m in members):
                    raise TypeError("mean_mlp members must be MeanProbe")
                self.heads_params[f"{key}_score"] = _stack(
                    [m.score.weight[0] for m in members]
                )
                self.heads_params[f"{key}_bias"] = _stack(
                    [m.score.bias[0] for m in members]
                )
                continue
            if not all(isinstance(m, HeadProbe) for m in members):
                raise TypeError(f"{name} members must be head probes")
            self.heads_params[f"{key}_value"] = _stack(
                [m.value.weight for m in members]
            )
            if members[0].query is not None:
                self.heads_params[f"{key}_query"] = _stack(
                    [m.query.weight for m in members]
                )
            self.heads_params[f"{key}_bias"] = _stack([m.bias for m in members])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return ``[batch, probes]`` logits in construction order."""
        batch, tokens, _ = x.shape
        first = torch.nn.functional.linear(x, self.first_weight, self.first_bias)
        out = x.new_zeros((batch, len(self.architectures)), dtype=torch.float32)
        n_linear = len(self.linear_index)
        if n_linear:
            scores = first[..., :n_linear].float().transpose(1, 2)  # [B, L, T]
            out[:, self.linear_index] = masked_mean(scores, mask.unsqueeze(1))
        if not self.deep_index:
            return out
        y = torch.relu(first[..., n_linear:]).reshape(batch * tokens, -1, WIDTH)
        y = torch.relu(
            torch.bmm(y.transpose(0, 1), self.second_weight.to(y.dtype)).transpose(0, 1)
            + self.second_bias.to(y.dtype)
        )  # [B*T, deep, WIDTH]
        for name, positions in self.groups.items():
            key = name.replace("-", "_")
            members = y[:, positions]  # [B*T, P, WIDTH]
            columns = [self.deep_index[p] for p in positions]
            if name == "mean_mlp":
                weight = self.heads_params[f"{key}_score"].to(members.dtype)
                scores = (members * weight).sum(-1).float()
                scores = scores.reshape(batch, tokens, -1).transpose(1, 2)
                pooled = masked_mean(scores, mask.unsqueeze(1))
                out[:, columns] = pooled + self.heads_params[f"{key}_bias"]
                continue
            value = self.heads_params[f"{key}_value"].to(members.dtype)
            count = len(positions)
            values = _project(members, value, batch, tokens)
            if name == "multimax":
                pooled = max_pool(values, mask)
            else:
                query = self.heads_params[f"{key}_query"].to(members.dtype)
                logits = _project(members, query, batch, tokens)
                if name == "attention":
                    pooled = attention_pool(logits, values, mask)
                else:
                    window = int(name.removeprefix("rolling_w"))
                    pooled = rolling_pool(logits, values, mask, window)
            pooled = pooled.reshape(batch, count, HEADS).sum(-1)
            out[:, columns] = pooled + self.heads_params[f"{key}_bias"]
        return out
