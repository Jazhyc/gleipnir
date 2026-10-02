"""Split logical example batches without changing optimizer-update boundaries."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MicrobatchPolicy:
    """A padded-token budget and a bounded power-of-two microbatch size."""

    max_padded_tokens: int
    max_micro_batch_size: int

    def __post_init__(self) -> None:
        if self.max_padded_tokens < 1:
            raise ValueError("microbatch token budget must be positive")
        size = self.max_micro_batch_size
        if size < 1 or size & (size - 1):
            raise ValueError("maximum microbatch size must be a positive power of two")

    def partition(self, lengths: Sequence[int]) -> list[list[int]]:
        """Sort within one update; permit over-budget sequences only as singletons."""
        if not lengths or any(length < 1 for length in lengths):
            raise ValueError("a logical batch needs positive sequence lengths")
        order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
        batches = []
        offset = 0
        while offset < len(order):
            width = lengths[order[offset]]
            limit = min(
                self.max_micro_batch_size,
                max(1, self.max_padded_tokens // width),
                len(order) - offset,
            )
            size = 1 << (limit.bit_length() - 1)
            batches.append(order[offset : offset + size])
            offset += size
        return batches


@dataclass
class LogicalBatch:
    """Keep unpadded CPU features opaque to Accelerate's recursive device transfer."""

    features: list[dict[str, Any]]


class LogicalBatchCollator:
    """Defer tensor construction until each physical microbatch is executed."""

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        return {"logical_batch": LogicalBatch(features)}


@contextmanager
def diagnostic_decoder_overrides(model, *, fp32_head: bool, maskless: bool):
    """Temporarily isolate head precision and right-padding masking in a canary."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    decoder, head = base.model, base.lm_head
    decoder_forward, head_forward = decoder.forward, head.forward

    def project(hidden):
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            bias = head.bias.float() if head.bias is not None else None
            return F.linear(hidden.float(), head.weight.float(), bias)

    def decode(*args, **kwargs):
        mask = kwargs.get("attention_mask")
        if mask is not None and (
            mask.ndim != 2 or bool((mask[:, 1:] > mask[:, :-1]).any())
        ):
            raise ValueError("maskless diagnostics require causal right padding")
        kwargs["attention_mask"] = None
        return decoder_forward(*args, **kwargs)

    try:
        if fp32_head:
            head.forward = project
        if maskless:
            decoder.forward = decode
        yield
    finally:
        decoder.forward, head.forward = decoder_forward, head_forward


class AdaptiveMicrobatchTrainerMixin:
    """Reuse Trainer backward/optimizer machinery with one logical batch per step."""

    def enable_adaptive_microbatching(
        self,
        policy: MicrobatchPolicy,
        physical_collator: Callable,
        *,
        profile: bool,
        require_finite_gradients: bool = False,
    ) -> None:
        if self.args.world_size != 1 or self.args.n_gpu > 1:
            raise ValueError("adaptive microbatching currently requires one device")
        if self.args.gradient_accumulation_steps != 1:
            raise ValueError(
                "adaptive microbatching requires one logical batch per step"
            )
        if self.model_accepts_loss_kwargs or self.compute_loss_func is not None:
            raise ValueError(
                "adaptive microbatching requires explicit per-example means"
            )
        if self.eval_dataset is not None:
            raise ValueError("adaptive microbatching currently supports training only")
        self.microbatch_policy = policy
        self.physical_collator = physical_collator
        self.data_collator = LogicalBatchCollator()
        self.microbatch_profile = profile
        self.require_finite_gradients = require_finite_gradients
        self.microbatch_records: list[dict[str, Any]] = []
        self.logical_batch_sizes: list[int] = []
        self._microbatch_loss_weight = 1.0
        self.microbatch_peak_allocated = 0
        self.microbatch_peak_reserved = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        result = super().compute_loss(
            model, inputs, return_outputs=return_outputs, **kwargs
        )
        weight = getattr(self, "_microbatch_loss_weight", 1.0)
        if return_outputs:
            loss, outputs = result
            return loss * weight, outputs
        return result * weight

    def training_step(self, model, inputs, num_items_in_batch=None):
        if self.current_gradient_accumulation_steps != 1:
            raise ValueError("adaptive optimizer boundary drifted")
        batch = inputs["logical_batch"]
        if not isinstance(batch, LogicalBatch):
            raise TypeError("expected an unpadded logical batch")
        features = batch.features
        lengths = [len(feature["direct_input_ids"]) for feature in features]
        self.logical_batch_sizes.append(len(features))
        total_loss = None
        cuda = self.args.device.type == "cuda"
        for indices in self.microbatch_policy.partition(lengths):
            selected = [features[i] for i in indices]
            if cuda and self.microbatch_profile:
                torch.cuda.synchronize(self.args.device)
                self.microbatch_peak_allocated = max(
                    self.microbatch_peak_allocated,
                    torch.cuda.max_memory_allocated(self.args.device),
                )
                self.microbatch_peak_reserved = max(
                    self.microbatch_peak_reserved,
                    torch.cuda.max_memory_reserved(self.args.device),
                )
                torch.cuda.reset_peak_memory_stats(self.args.device)
            started = time.perf_counter()
            self._microbatch_loss_weight = len(indices) / len(features)
            try:
                loss = super().training_step(
                    model, self.physical_collator(selected), None
                )
            finally:
                self._microbatch_loss_weight = 1.0
            if cuda and self.microbatch_profile:
                torch.cuda.synchronize(self.args.device)
            elapsed = time.perf_counter() - started
            allocated = torch.cuda.max_memory_allocated(self.args.device) if cuda else 0
            reserved = torch.cuda.max_memory_reserved(self.args.device) if cuda else 0
            self.microbatch_peak_allocated = max(
                self.microbatch_peak_allocated, allocated
            )
            self.microbatch_peak_reserved = max(self.microbatch_peak_reserved, reserved)
            self.microbatch_records.append(
                {
                    "update": len(self.logical_batch_sizes),
                    "examples": len(indices),
                    "logical_indices": indices,
                    "max_length": max(lengths[i] for i in indices),
                    "tokens": sum(lengths[i] for i in indices),
                    "padded_tokens": len(indices) * max(lengths[i] for i in indices),
                    "seconds": elapsed,
                    "peak_allocated_bytes": allocated,
                    "peak_reserved_bytes": reserved,
                }
            )
            total_loss = loss if total_loss is None else total_loss + loss
        if not torch.isfinite(total_loss).item():
            raise FloatingPointError("nonfinite adaptive logical-batch loss")
        if self.require_finite_gradients:
            gradients = [p.grad for p in model.parameters() if p.requires_grad]
            if not gradients or any(g is None for g in gradients):
                raise FloatingPointError("missing adaptive adapter gradient")
            torch.nn.utils.get_total_norm(gradients, error_if_nonfinite=True)
        return total_loss

    def adaptive_microbatch_metadata(self) -> dict[str, Any]:
        """Record partitions, per-example normalization, and profiling mode."""
        return {
            "enabled": True,
            "policy": asdict(self.microbatch_policy),
            "loss_normalization": "sum_per_example_over_logical_batch_v1",
            "logical_batch_sizes": self.logical_batch_sizes,
            "physical_microbatch_sizes": sorted(
                {r["examples"] for r in self.microbatch_records}
            ),
            "require_finite_gradients": self.require_finite_gradients,
            "profiling": "synchronized"
            if self.microbatch_profile
            else "unsynchronized_wall",
            "records": self.microbatch_records,
        }


def gradient_partition_canary(
    model: torch.nn.Module,
    features: list[dict[str, Any]],
    collator: Callable,
    loss_forward: Callable,
    policy: MicrobatchPolicy,
    *,
    relative_tolerance: float = 0.05,
) -> dict[str, Any]:
    """Compare all trainable gradients for singleton versus adaptive partitions."""
    named_parameters = [
        (name, p) for name, p in model.named_parameters() if p.requires_grad
    ]
    parameters = [p for _, p in named_parameters]
    was_training = model.training
    reference = []
    reference_losses = []
    actual_losses = []
    try:
        model.train()
        model.zero_grad(set_to_none=True)
        for feature in features:
            loss = loss_forward(collator([feature]))
            reference_losses.append(float(loss.detach()))
            (loss / len(features)).backward()
        reference = [
            None if p.grad is None else p.grad.detach().cpu().clone()
            for p in parameters
        ]
        model.zero_grad(set_to_none=True)
        lengths = [len(f["direct_input_ids"]) for f in features]
        partition = policy.partition(lengths)
        for indices in partition:
            loss = loss_forward(collator([features[i] for i in indices]))
            actual_losses.append(float(loss.detach()) * len(indices) / len(features))
            (loss * (len(indices) / len(features))).backward()
        norm_squared = error_squared = 0.0
        actual_norm_squared = dot_product = 0.0
        maximum_error = 0.0
        compared = 0
        differences = []
        for (name, parameter), ref in zip(named_parameters, reference, strict=True):
            if (ref is None) != (parameter.grad is None):
                raise ValueError(
                    "gradient partition changed which parameters receive gradients"
                )
            if ref is None:
                continue
            actual = parameter.grad.detach().float().cpu()
            difference = actual - ref.float()
            ref_norm = float(ref.float().square().sum())
            error_norm = float(difference.square().sum())
            norm_squared += ref_norm
            error_squared += error_norm
            actual_norm_squared += float(actual.square().sum())
            dot_product += float((actual * ref.float()).sum())
            differences.append(
                {
                    "parameter": name,
                    "error_norm": math.sqrt(error_norm),
                    "reference_norm": math.sqrt(ref_norm),
                }
            )
            maximum_error = max(maximum_error, float(difference.abs().max()))
            compared += ref.numel()
        relative_error = (
            math.sqrt(error_squared / norm_squared) if norm_squared > 0 else math.inf
        )
        return {
            "finite": all(math.isfinite(x) for x in reference_losses + actual_losses)
            and math.isfinite(norm_squared)
            and norm_squared > 0
            and math.isfinite(actual_norm_squared)
            and actual_norm_squared > 0
            and all(x is not None for x in reference)
            and all(p.grad is not None for p in parameters),
            "passed": math.isfinite(relative_error)
            and relative_error <= relative_tolerance,
            "relative_l2_error": relative_error,
            "relative_tolerance": relative_tolerance,
            "maximum_absolute_error": maximum_error,
            "compared_parameters": compared,
            "reference_gradient_norm": math.sqrt(norm_squared),
            "actual_gradient_norm": math.sqrt(actual_norm_squared),
            "gradient_cosine_similarity": (
                dot_product / math.sqrt(norm_squared * actual_norm_squared)
                if norm_squared > 0 and actual_norm_squared > 0
                else None
            ),
            "reference_mean_loss": sum(reference_losses) / len(features),
            "actual_mean_loss": sum(actual_losses),
            "largest_gradient_differences": sorted(
                differences, key=lambda d: -d["error_norm"]
            )[:8],
            "examples": len(features),
            "physical_microbatch_sizes": [len(x) for x in partition],
        }
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)
