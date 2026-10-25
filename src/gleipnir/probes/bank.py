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
    AlphaEvolveProbe,
    HeadProbe,
    MeanProbe,
    alphaevolve_penalty,
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
        evolved = [
            i for i, a in enumerate(architectures) if a.startswith("alphaevolve")
        ]
        deep = [i for i in range(len(probes)) if i not in set(linear) | set(evolved)]
        self.linear_index, self.deep_index = linear, deep
        self._init_alphaevolve([probes[i] for i in evolved], evolved)
        first_w, first_b = [], []
        for i in linear:
            first_w.append(probes[i].score.weight)  # [1, hidden]
            first_b.append(probes[i].score.bias)
        for i in deep:
            first_w.append(probes[i].body[0].weight)  # [WIDTH, hidden]
            first_b.append(probes[i].body[0].bias)
        if first_w:
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
        out = x.new_zeros((batch, len(self.architectures)), dtype=torch.float32)
        if self.evolved_index:
            out[:, self.evolved_index] = self._alphaevolve(x, mask)
        if not (self.linear_index or self.deep_index):
            return out
        first = torch.nn.functional.linear(x, self.first_weight, self.first_bias)
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

    def _init_alphaevolve(self, members: list[nn.Module], index: list[int]) -> None:
        self.evolved_index = index
        if not members:
            return
        if not all(isinstance(m, AlphaEvolveProbe) for m in members):
            raise TypeError("alphaevolve members must be AlphaEvolveProbe")
        stacks = {
            "norm_weight": [m.norm.weight for m in members],
            "norm_bias": [m.norm.bias for m in members],
            "first_weight": [m.body[0].weight for m in members],  # [WIDTH, hidden]
            "first_bias": [m.body[0].bias for m in members],
            "second_weight": [m.body[2].weight.T for m in members],
            "second_bias": [m.body[2].bias for m in members],
            "project": [m.project.weight for m in members],  # [channels, WIDTH]
            "gate": [m.gate.weight for m in members],
            "out_weight": [m.out.weight[0] for m in members],
            "out_bias": [m.out.bias[0] for m in members],
        }
        self.evolved = nn.ParameterDict({k: _stack(v) for k, v in stacks.items()})
        self.register_buffer("evolved_l1", torch.tensor([m.l1 for m in members]))
        self.register_buffer("evolved_ortho", torch.tensor([m.ortho for m in members]))

    def _alphaevolve(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``[B, P]`` logits; each probe's LayerNorm affine is folded into its first
        layer so all probes share one normalized input and one matmul."""
        e = self.evolved
        batch, tokens, hidden = x.shape
        count = e["first_weight"].shape[0]
        normed = torch.nn.functional.layer_norm(x.float(), (hidden,)).to(x.dtype)
        weight = e["first_weight"] * e["norm_weight"].unsqueeze(1)
        bias = e["first_bias"] + torch.einsum(
            "pwh,ph->pw", e["first_weight"], e["norm_bias"]
        )
        first = torch.nn.functional.linear(
            normed, weight.reshape(-1, hidden), bias.reshape(-1)
        )
        y = torch.relu(first).reshape(batch * tokens, count, WIDTH)
        y = torch.relu(
            torch.bmm(y.transpose(0, 1), e["second_weight"].to(y.dtype)).transpose(0, 1)
            + e["second_bias"].to(y.dtype)
        )
        project = torch.einsum("npw,pkw->npk", y, e["project"].to(y.dtype))
        gate = torch.einsum("npw,pkw->npk", y, e["gate"].to(y.dtype))
        values = (project * torch.nn.functional.softplus(gate)).float()
        values = values.reshape(batch, tokens, -1).transpose(1, 2)  # [B, P*K, T]
        channels = e["project"].shape[1]
        top = max_pool(values, mask).reshape(batch, count, channels)
        bottom = max_pool(-values, mask).reshape(batch, count, channels)
        pooled = torch.cat([top, bottom], -1)  # [B, P, 2K]
        return torch.einsum("bpc,pc->bp", pooled, e["out_weight"]) + e["out_bias"]

    def regularization(self) -> torch.Tensor:
        """``[probes]`` extra training penalties (zero except AlphaEvolve probes)."""
        device = next(self.parameters()).device
        penalty = torch.zeros(len(self.architectures), device=device)
        if not self.evolved_index:
            return penalty
        e = self.evolved
        for j, i in enumerate(self.evolved_index):
            penalty[i] = alphaevolve_penalty(
                [p[j] for p in e.values()],
                e["project"][j],
                float(self.evolved_l1[j]),
                float(self.evolved_ortho[j]),
            )
        return penalty
