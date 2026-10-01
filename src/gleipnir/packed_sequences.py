"""Explicit packed boundaries for bounded isolation diagnostics.

These utilities do not enable packing in the training runner. Native recurrent
and convolution kernels must honor the supplied boundaries independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import accumulate

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
