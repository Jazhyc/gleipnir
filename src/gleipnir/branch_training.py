"""Eager, differentiable shared-prefix execution for text-only Qwen3.5."""

import copy
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer, LinearAttentionLayer


class BranchCache(DynamicCache):
    """Copy-on-update recurrent state; fork containers but preserve autograd edges."""

    def update_conv_state(
        self,
        conv_states: torch.Tensor,
        layer_idx: int,
        state_idx: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        layer = self.layers[layer_idx]
        if type(layer) is not LinearAttentionLayer:
            raise TypeError("unsupported recurrent cache layout")
        # Qwen supplies its complete trailing convolution window, already padded.
        layer.conv_states[state_idx] = conv_states
        layer.conv_kernel_size[state_idx] = conv_states.shape[-1]
        layer.has_previous_state[state_idx] = True
        layer.is_conv_states_initialized[state_idx] = True
        layer.dtype, layer.device = conv_states.dtype, conv_states.device
        return conv_states

    def update_recurrent_state(
        self,
        recurrent_states: torch.Tensor,
        layer_idx: int,
        state_idx: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        layer = self.layers[layer_idx]
        if type(layer) is not LinearAttentionLayer:
            raise TypeError("unsupported recurrent cache layout")
        layer.recurrent_states[state_idx] = recurrent_states
        layer.is_recurrent_states_initialized[state_idx] = True
        return recurrent_states

    def fork(self) -> "BranchCache":
        """Share immutable tensor values, never mutable layer/state containers."""
        result = copy.copy(self)
        result.layers = []
        for layer in self.layers:
            if type(layer) not in (DynamicLayer, LinearAttentionLayer):
                raise TypeError("unsupported attention cache layout")
            cloned = copy.copy(layer)
            for name, value in vars(layer).items():
                if isinstance(value, (dict, list)):
                    setattr(cloned, name, copy.copy(value))
            result.layers.append(cloned)
        return result


@dataclass(frozen=True)
class BranchPlan:
    """Requests are in original order; trunk is the final full request."""

    requests: tuple[tuple[int, ...], ...]
    split_positions: tuple[int, ...]
    alignment: int = 1

    @property
    def processed_tokens(self) -> int:
        return self.token_work()

    def token_work(self, *, independent_endpoint: bool = False) -> int:
        if independent_endpoint:
            return (
                max(self.split_positions[:-1], default=0)
                + sum(
                    len(row) - split
                    for row, split in zip(
                        self.requests[:-1], self.split_positions[:-1], strict=True
                    )
                )
                + len(self.requests[-1])
            )
        return max(self.split_positions) + sum(
            len(row) - split
            for row, split in zip(self.requests, self.split_positions, strict=True)
        )


def plan_branches(
    requests: Sequence[Sequence[int]], *, alignment: int = 1
) -> BranchPlan:
    """Preserve exact token sequences, allowing two-token-or-longer continuations."""
    rows = tuple(tuple(int(token) for token in row) for row in requests)
    if alignment not in (1, 64):
        raise ValueError("unsupported branch alignment")
    if not rows or any(len(row) < 2 for row in rows):
        raise ValueError("each request needs at least two tokens")
    trunk = rows[-1]
    splits = []
    for row in rows:
        common = 0
        for left, right in zip(row, trunk, strict=False):
            if left != right:
                break
            common += 1
        split = min(common, len(row) - 2, len(trunk) - 2)
        splits.append(split // alignment * alignment)
    # Cached single-token continuations use inference-only kernels. Coalesce
    # adjacent splits instead; their branch suffix simply includes one more token.
    mapping = {}
    previous = 0
    for split in sorted(set(splits)):
        safe = previous if split - previous < 2 else split
        mapping[split] = safe
        previous = safe
    return BranchPlan(rows, tuple(mapping[s] for s in splits), alignment)


def branched_decision_logits(
    model: torch.nn.Module,
    plan: BranchPlan,
    decision_ids: Sequence[int],
    *,
    checkpoint_segments: bool = False,
    fp32_head: bool = False,
    independent_endpoint: bool = False,
) -> torch.Tensor:
    """Return one decision-logit vector per request, preserving shared gradients.

    Call backward once on the combined parent objective before any optimizer
    update. No cache survives this function's computation graph or training step.
    The initial prototype intentionally does not support padding or batching.
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if plan != plan_branches(plan.requests, alignment=plan.alignment):
        raise ValueError("invalid or unsafe branch plan")
    if type(base).__name__ != "Qwen3_5ForCausalLM":
        raise TypeError("branch prototype supports text-only Qwen3.5 causal LM")
    if any(
        getattr(module, "gradient_checkpointing", False) for module in model.modules()
    ):
        raise ValueError(
            "branch cache requires functional checkpointing; disabled here"
        )
    if any(
        isinstance(module, torch.nn.Dropout) and module.p for module in model.modules()
    ):
        raise ValueError("dropout requires an explicit shared-randomness policy")
    if getattr(base.config, "attention_dropout", 0):
        raise ValueError("attention dropout is unsupported")
    device = base.get_input_embeddings().weight.device
    cache = BranchCache(config=base.config)

    def decision(hidden):
        head = base.get_output_embeddings()
        if type(head) is not torch.nn.Linear:
            raise TypeError("selected projection requires an unwrapped linear head")
        ids = torch.tensor(decision_ids, device=head.weight.device)
        weight = head.weight.index_select(0, ids)
        bias = None if head.bias is None else head.bias.index_select(0, ids)
        if fp32_head:
            with torch.autocast("cuda", enabled=False):
                return torch.nn.functional.linear(
                    hidden.to(weight.device).float(),
                    weight.float(),
                    None if bias is None else bias.float(),
                )[0]
        return torch.nn.functional.linear(hidden.to(weight.device), weight, bias)[0]

    def segment(tokens, source):
        # Every invocation, including checkpoint recomputation, gets a private
        # mutable container. Captured source tensors are immutable graph edges.
        def forward(ids):
            destination = source.fork()
            hidden = base.model(
                input_ids=ids, past_key_values=destination, use_cache=True
            ).last_hidden_state
            return hidden, destination

        if checkpoint_segments:
            from torch.utils.checkpoint import checkpoint

            return checkpoint(forward, tokens, use_reentrant=False)
        return forward(tokens)

    result = [None] * len(plan.requests)
    position = 0
    shared_splits = (
        plan.split_positions[:-1] if independent_endpoint else plan.split_positions
    )
    for split in sorted(set(shared_splits)):
        if split > position:
            tokens = torch.tensor([plan.requests[-1][position:split]], device=device)
            _, cache = segment(tokens, cache)
            position = split
        for index, branch_split in enumerate(plan.split_positions):
            if independent_endpoint and index == len(plan.requests) - 1:
                continue
            if branch_split != split:
                continue
            tokens = torch.tensor([plan.requests[index][split:]], device=device)
            hidden, _ = segment(tokens, cache)
            hidden = hidden[:, -1, :]
            result[index] = decision(hidden)
    if independent_endpoint:
        if checkpoint_segments:
            from functools import partial

            from torch.utils.checkpoint import checkpoint

            # The public enable method installs an embedding-grad hook which
            # survives disable() on this Transformers version. That would alter
            # the earlier shared forward during checkpoint recomputation.
            base._set_gradient_checkpointing(
                enable=True,
                gradient_checkpointing_func=partial(checkpoint, use_reentrant=False),
            )
        try:
            tokens = torch.tensor([plan.requests[-1]], device=device)
            hidden = base.model(input_ids=tokens, use_cache=False).last_hidden_state[
                :, -1, :
            ]
            result[-1] = decision(hidden)
        finally:
            if checkpoint_segments:
                base._set_gradient_checkpointing(enable=False)
    return torch.stack(result)
