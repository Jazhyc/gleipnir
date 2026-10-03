"""Explicit packed boundaries for bounded isolation diagnostics.

Packing is enabled only by the bounded BF16 training screen. Native recurrent
and convolution kernels must honor the supplied boundaries independently.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import accumulate
from typing import Any

import torch


@dataclass(frozen=True)
class PackedSequenceLayout:
    """Describe independent, nonempty sequences flattened into one batch row."""

    lengths: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.lengths or any(
            not isinstance(length, int) or isinstance(length, bool) or length < 1
            for length in self.lengths
        ):
            raise ValueError("packed sequences require positive integer lengths")
        if self.total_tokens >= 2**31:
            raise ValueError("packed cumulative lengths exceed the int32 contract")

    @property
    def total_tokens(self) -> int:
        return sum(self.lengths)

    @property
    def offsets(self) -> tuple[int, ...]:
        return (0, *accumulate(self.lengths))

    def kernel_kwargs(self, device: torch.device | str = "cpu") -> dict:
        """Use one boundary source for positions, convolution and recurrence."""
        cumulative = torch.tensor(self.offsets, dtype=torch.int32, device=device)
        positions = torch.cat(
            [torch.arange(length, device=device) for length in self.lengths]
        ).unsqueeze(0)
        sequence_ids = torch.repeat_interleave(
            torch.arange(len(self.lengths), dtype=torch.int32, device=device),
            torch.tensor(self.lengths, device=device),
        ).unsqueeze(0)
        return {
            "position_ids": positions,
            "seq_idx": sequence_ids,
            "cu_seq_lens_q": cumulative,
            "cu_seq_lens_k": cumulative,
            "max_length_q": max(self.lengths),
            "max_length_k": max(self.lengths),
            "use_cache": False,
        }

    def decision_positions(self, device: torch.device | str = "cpu") -> torch.Tensor:
        """Return one last-input-token index per original monitoring example."""
        return torch.tensor(self.offsets[1:], dtype=torch.long, device=device) - 1

    def dense_causal_mask(
        self, device: torch.device | str = "cpu", *, max_tokens: int = 2048
    ) -> torch.Tensor:
        """Build an SDPA boolean oracle; True means allowed, including the diagonal.

        Bound allocation explicitly: this quadratic mask is a short correctness
        oracle, not the long-context performance implementation.
        """
        if max_tokens < 1 or self.total_tokens > max_tokens:
            raise ValueError("dense packed mask exceeds the diagnostic token limit")
        sequence_ids = self.kernel_kwargs(device)["seq_idx"][0]
        indices = torch.arange(self.total_tokens, device=device)
        same_sequence = sequence_ids[:, None] == sequence_ids[None, :]
        causal = indices[:, None] >= indices[None, :]
        return (same_sequence & causal)[None, None, :, :]

    def sdpa_mask_mapping(self, device: torch.device | str = "cpu") -> dict:
        """Keep attention isolation separate from the recurrent padding signal."""
        return {
            "full_attention": self.dense_causal_mask(device),
            "linear_attention": None,
        }


def collate_packed_monitoring(features: Sequence[dict[str, Any]]) -> dict:
    """Flatten direct binary monitoring prompts, retaining every example target."""
    if not features or any(
        "input_ids" in f
        or "mil_positions" in f
        or "prefix_input_ids" in f
        or "soft_rating_probs" in f
        for f in features
    ):
        raise ValueError("packing supports direct binary monitoring only")
    lengths = tuple(len(f["direct_input_ids"]) for f in features)
    PackedSequenceLayout(lengths)
    batch = {
        "direct_input_ids": torch.tensor(
            [[token for f in features for token in f["direct_input_ids"]]],
            dtype=torch.long,
        ),
        "packed_lengths": lengths,
        "binary_labels": torch.tensor([f["binary_label"] for f in features]),
        "dataset_ids": torch.tensor([f["dataset_id"] for f in features]),
    }
    if all("soft_target" in f for f in features):
        batch["soft_targets"] = torch.tensor(
            [f["soft_target"] for f in features], dtype=torch.float32
        )
    elif any("soft_target" in f for f in features):
        raise ValueError("mixed missing soft targets in packed batch")
    from gleipnir.binary_task_training import collate_binary_task_fields

    batch.update(collate_binary_task_fields(features))
    return batch


def packed_partition(lengths: Sequence[int], budget: int) -> list[list[int]]:
    """Best-fit complete examples within one logical update; retain long singletons."""
    PackedSequenceLayout(tuple(lengths))
    if budget < 1:
        raise ValueError("packing token budget must be positive")
    groups: list[list[int]] = []
    totals: list[int] = []
    for i in sorted(range(len(lengths)), key=lambda j: -lengths[j]):
        fits = [j for j, total in enumerate(totals) if total + lengths[i] <= budget]
        if fits:
            j = max(fits, key=lambda j: totals[j])
            groups[j].append(i)
            totals[j] += lengths[i]
        else:
            groups.append([i])
            totals.append(lengths[i])
    return groups


def segmented_sdpa_interface(original: Callable) -> Callable:
    """Dispatch unpadded attention independently inside each original sequence."""

    def attention(module, query, key, value, attention_mask, **kwargs):
        cumulative = kwargs.get("cu_seq_lens_q")
        if cumulative is None:
            return original(module, query, key, value, attention_mask, **kwargs)
        if query.shape[0] != 1 or attention_mask is not None:
            raise ValueError("segmented SDPA requires one unpadded flattened row")
        if not torch.equal(cumulative, kwargs["cu_seq_lens_k"]):
            raise ValueError("packed self attention requires matching Q/K boundaries")
        cuts = cumulative.tolist()
        if cuts[0] != 0 or cuts[-1] != query.shape[2]:
            raise ValueError("attention boundaries do not cover the input")
        options = {
            k: v
            for k, v in kwargs.items()
            if k
            not in {
                "cu_seq_lens_q",
                "cu_seq_lens_k",
                "max_length_q",
                "max_length_k",
                "position_ids",
                "seq_idx",
            }
        }
        options["is_causal"] = True
        outputs = []
        for start, end in zip(cuts[:-1], cuts[1:], strict=True):
            if end <= start:
                raise ValueError("empty packed attention sequence")
            output, _ = original(
                module,
                query[:, :, start:end],
                key[:, :, start:end],
                value[:, :, start:end],
                None,
                **options,
            )
            outputs.append(output)
        return torch.cat(outputs, dim=1), None

    return attention


@contextmanager
def installed_segmented_sdpa():
    """Keep the native router opaque to compilation, restoring its global binding."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    router = torch.compiler.disable(segmented_sdpa_interface(original))
    router._gleipnir_packed_boundaries = True
    ALL_ATTENTION_FUNCTIONS.register("sdpa", router)
    try:
        yield
    finally:
        ALL_ATTENTION_FUNCTIONS.register("sdpa", original)


def forward_packed_monitoring_logits(
    model: Any,
    input_ids: torch.Tensor,
    lengths: tuple[int, ...],
    *,
    inputs_embeds: torch.Tensor | None = None,
) -> tuple[torch.Tensor, Any]:
    """Select every example's decision boundary without projecting all tokens."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    if not getattr(
        ALL_ATTENTION_FUNCTIONS["sdpa"], "_gleipnir_packed_boundaries", False
    ):
        raise ValueError("packed readout requires the isolated SDPA router")
    layout = PackedSequenceLayout(lengths)
    if input_ids.shape != (1, layout.total_tokens):
        raise ValueError("packed input shape disagrees with sequence lengths")
    kwargs = layout.kernel_kwargs(input_ids.device)
    kwargs["attention_mask"] = {"full_attention": None, "linear_attention": None}
    kwargs["logits_to_keep"] = layout.decision_positions(input_ids.device)
    if inputs_embeds is None:
        kwargs["input_ids"] = input_ids
    else:
        kwargs["inputs_embeds"] = inputs_embeds
    outputs = model(**kwargs)
    if outputs.logits.shape[:2] != (1, len(lengths)):
        raise ValueError("packed model did not preserve every decision boundary")
    return outputs.logits[0], outputs
