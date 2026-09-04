#!/usr/bin/env python3
"""SFT a Qwen LoRA on hard, rationale, or soft monitor targets."""

from __future__ import annotations

import bisect
import hashlib
import json
import os
import random
import re
import time
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, SequentialSampler, WeightedRandomSampler

from gleipnir.qwen35_fast_training import prewarm_gated_delta_rule
from gleipnir.qwen35_loftq import initialize_qwen35_loftq
from gleipnir.training import (
    MuonAdamW,
    muon_adamw_param_groups,
)


class CompletionOnlyCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id
        self.batches = 0
        self.examples = 0
        self.input_tokens = 0
        self.input_padded_tokens = 0
        self.direct_tokens = 0
        self.direct_padded_tokens = 0

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        self.batches += 1
        self.examples += len(features)
        batch = {}
        if "input_ids" in features[0]:
            width = max(len(feature["input_ids"]) for feature in features)
            self.input_tokens += sum(len(feature["input_ids"]) for feature in features)
            self.input_padded_tokens += width * len(features)
            input_ids, attention_mask, labels = [], [], []
            for feature in features:
                padding = width - len(feature["input_ids"])
                input_ids.append(feature["input_ids"] + [self.pad_token_id] * padding)
                attention_mask.append([1] * len(feature["input_ids"]) + [0] * padding)
                labels.append(feature["labels"] + [-100] * padding)
            batch.update(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            )
        if "direct_input_ids" in features[0]:
            direct_width = max(len(feature["direct_input_ids"]) for feature in features)
            self.direct_tokens += sum(
                len(feature["direct_input_ids"]) for feature in features
            )
            self.direct_padded_tokens += direct_width * len(features)
            direct_input_ids, direct_attention_mask = [], []
            for feature in features:
                padding = direct_width - len(feature["direct_input_ids"])
                direct_input_ids.append(
                    feature["direct_input_ids"] + [self.pad_token_id] * padding
                )
                direct_attention_mask.append(
                    [1] * len(feature["direct_input_ids"]) + [0] * padding
                )
            batch.update(
                {
                    "direct_input_ids": torch.tensor(
                        direct_input_ids, dtype=torch.long
                    ),
                    "direct_attention_mask": torch.tensor(
                        direct_attention_mask, dtype=torch.long
                    ),
                    "binary_labels": torch.tensor(
                        [feature["binary_label"] for feature in features],
                        dtype=torch.long,
                    ),
                    "dataset_ids": torch.tensor(
                        [feature["dataset_id"] for feature in features],
                        dtype=torch.long,
                    ),
                }
            )
            if "soft_target" in features[0]:
                batch["soft_targets"] = torch.tensor(
                    [feature["soft_target"] for feature in features],
                    dtype=torch.float32,
                )
            if "soft_rating_probs" in features[0]:
                batch["soft_rating_targets"] = torch.tensor(
                    [feature["soft_rating_probs"] for feature in features],
                    dtype=torch.float32,
                )
            if "mil_positions" in features[0]:
                mil_width = max(len(feature["mil_positions"]) for feature in features)
                positions, position_mask = [], []
                for feature in features:
                    values = list(feature["mil_positions"])
                    padding = mil_width - len(values)
                    positions.append(values + [0] * padding)
                    position_mask.append([True] * len(values) + [False] * padding)
                batch["mil_positions"] = torch.tensor(positions, dtype=torch.long)
                batch["mil_position_mask"] = torch.tensor(
                    position_mask, dtype=torch.bool
                )
        return batch

    def padding_statistics(self) -> dict[str, int | float | None]:
        """Return observed padding work for the batches emitted so far."""

        def fraction(tokens: int, padded_tokens: int) -> float | None:
            if padded_tokens == 0:
                return None
            return 1.0 - tokens / padded_tokens

        return {
            "batches": self.batches,
            "examples": self.examples,
            "input_tokens": self.input_tokens,
            "input_padded_tokens": self.input_padded_tokens,
            "input_padding_fraction": fraction(
                self.input_tokens, self.input_padded_tokens
            ),
            "direct_tokens": self.direct_tokens,
            "direct_padded_tokens": self.direct_padded_tokens,
            "direct_padding_fraction": fraction(
                self.direct_tokens, self.direct_padded_tokens
            ),
        }


def apply_gradient_checkpointing_policy(
    model: torch.nn.Module,
    policy: str,
    layer_indices: Sequence[int] | None = None,
) -> list[int]:
    """Checkpoint all, all linear-attention, or explicit decoder layers."""
    if policy not in {"all", "linear_attention_only", "explicit"}:
        raise ValueError(
            "student.training.gradient_checkpointing_policy must be all, "
            "linear_attention_only, or explicit"
        )
    if policy == "explicit" and layer_indices is None:
        raise ValueError("explicit checkpointing requires layer indices")
    if policy != "explicit" and layer_indices is not None:
        raise ValueError("checkpointing layer indices require the explicit policy")
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    text_model = getattr(base, "model", None)
    layers = getattr(text_model, "layers", None)
    layer_types = getattr(base.config, "layer_types", None)
    if layers is None or layer_types is None or len(layers) != len(layer_types):
        if policy != "all" or layer_indices is not None:
            raise ValueError("selective checkpointing requires Qwen layer metadata")
        return []
    requested = list(layer_indices or [])
    if len(set(requested)) != len(requested):
        raise ValueError("checkpointing layer indices must be unique")
    if any(index < 0 or index >= len(layers) for index in requested):
        raise ValueError("checkpointing layer index is out of range")
    if policy == "explicit" and any(
        layer_types[index] != "linear_attention" for index in requested
    ):
        raise ValueError("explicit checkpointing is restricted to linear attention")
    requested_set = set(requested)
    checkpointed = []
    for index, (layer, layer_type) in enumerate(zip(layers, layer_types, strict=True)):
        enabled = (
            policy == "all"
            or (policy == "linear_attention_only" and layer_type == "linear_attention")
            or (policy == "explicit" and index in requested_set)
        )
        layer.gradient_checkpointing = enabled
        if enabled:
            checkpointed.append(index)
    return checkpointed


def trainer_should_manage_gradient_checkpointing(enabled: bool, policy: str) -> bool:
    """Avoid Trainer resetting a selective per-layer checkpoint policy."""
    return enabled and policy == "all"


def apply_selective_torch_compile_policy(
    model: torch.nn.Module,
    policy: str,
    *,
    backend: str,
    mode: str,
    dynamic: bool,
    compile_function: Callable[..., Callable[..., Any]] = torch.compile,
    disable_function: Callable[..., Callable[..., Any]] = torch.compiler.disable,
) -> list[int]:
    """Compile selected Qwen layer shells while preserving custom kernels."""
    if policy not in {
        "none",
        "uncheckpointed_full_attention",
        "full_attention_and_linear_shell",
        "full_attention_and_linear_mixer_segments",
        "checkpointed_full_attention_and_linear_shell",
        "decoder_shells_without_token_mixers",
        "linear_attention_shells_only",
    }:
        raise ValueError(
            "student.training.selective_torch_compile_policy must be none, "
            "uncheckpointed_full_attention, full_attention_and_linear_shell, "
            "full_attention_and_linear_mixer_segments, or "
            "checkpointed_full_attention_and_linear_shell, or "
            "decoder_shells_without_token_mixers, or linear_attention_shells_only"
        )
    if policy == "none":
        return []
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    text_model = getattr(base, "model", None)
    layers = getattr(text_model, "layers", None)
    layer_types = getattr(base.config, "layer_types", None)
    if layers is None or layer_types is None or len(layers) != len(layer_types):
        raise ValueError("selective compilation requires Qwen layer metadata")
    state_keys_before = tuple(model.state_dict())
    compiled = []
    disabled_kernel_functions: dict[int, Callable[..., Any]] = {}
    for index, (layer, layer_type) in enumerate(zip(layers, layer_types, strict=True)):
        compile_full = (
            layer_type == "full_attention" and policy != "linear_attention_shells_only"
        )
        compile_linear_shell = (
            policy
            in {
                "full_attention_and_linear_shell",
                "full_attention_and_linear_mixer_segments",
                "checkpointed_full_attention_and_linear_shell",
                "decoder_shells_without_token_mixers",
                "linear_attention_shells_only",
            }
            and layer_type == "linear_attention"
        )
        if not compile_full and not compile_linear_shell:
            continue
        if (
            compile_full
            and bool(getattr(layer, "gradient_checkpointing", False))
            and policy != "checkpointed_full_attention_and_linear_shell"
        ):
            raise ValueError(
                "selective compilation requires full-attention layers to be "
                "uncheckpointed"
            )
        if compile_full and policy == "decoder_shells_without_token_mixers":
            self_attn = getattr(layer, "self_attn", None)
            if self_attn is None:
                raise ValueError("full-attention layer exposes no token mixer")
            self_attn.forward = disable_function(self_attn.forward)
        if compile_linear_shell:
            linear_attn = getattr(layer, "linear_attn", None)
            if linear_attn is None:
                raise ValueError("linear-attention layer exposes no token mixer")
            if policy in {
                "full_attention_and_linear_shell",
                "checkpointed_full_attention_and_linear_shell",
                "decoder_shells_without_token_mixers",
                "linear_attention_shells_only",
            }:
                linear_attn.forward = disable_function(linear_attn.forward)
            else:
                for attribute in ("causal_conv1d_fn", "chunk_gated_delta_rule"):
                    kernel_function = getattr(linear_attn, attribute, None)
                    if kernel_function is None:
                        raise ValueError(
                            f"linear-attention layer exposes no {attribute} kernel"
                        )
                    identity = id(kernel_function)
                    disabled = disabled_kernel_functions.get(identity)
                    if disabled is None:
                        disabled = disable_function(kernel_function)
                        disabled_kernel_functions[identity] = disabled
                    setattr(linear_attn, attribute, disabled)
                gated_norm = getattr(linear_attn, "norm", None)
                if gated_norm is None:
                    raise ValueError("linear-attention layer exposes no gated norm")
                gated_norm.forward = disable_function(gated_norm.forward)
        layer.forward = compile_function(
            layer.forward,
            backend=backend,
            mode=mode,
            dynamic=dynamic,
            fullgraph=False,
        )
        compiled.append(index)
    state_keys_after = tuple(model.state_dict())
    if state_keys_after != state_keys_before or any(
        "_orig_mod" in key for key in state_keys_after
    ):
        raise RuntimeError("selective compilation changed model state-dict keys")
    return compiled


def torch_compile_counter_snapshot() -> dict[str, dict[str, int | float]]:
    """Return JSON-safe Dynamo counters after an optional compiled run."""
    counters = getattr(getattr(torch, "_dynamo", None), "utils", None)
    counters = getattr(counters, "counters", {})
    return {
        str(category): {
            str(name): value
            for name, value in values.items()
            if isinstance(value, (int, float))
        }
        for category, values in counters.items()
        if values
    }


def compare_compile_canary_logits(
    eager_logits: torch.Tensor,
    compiled_logits: torch.Tensor,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    """Compare selected decision logits under explicit mixed-precision bounds."""
    if eager_logits.shape != compiled_logits.shape:
        raise ValueError("eager and compiled canary logits have different shapes")
    eager = eager_logits.detach().float().cpu()
    compiled = compiled_logits.detach().float().cpu()
    differences = (compiled - eager).abs()
    tolerances = absolute_tolerance + relative_tolerance * torch.maximum(
        eager.abs(), compiled.abs()
    )
    eager_margin = float(eager[0, 1] - eager[0, 0])
    compiled_margin = float(compiled[0, 1] - compiled[0, 0])
    return {
        "eager_logits": eager.tolist(),
        "compiled_logits": compiled.tolist(),
        "maximum_absolute_difference": float(differences.max()),
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "eager_margin": eager_margin,
        "compiled_margin": compiled_margin,
        "margin_absolute_difference": abs(compiled_margin - eager_margin),
        "passed": bool(torch.all(differences <= tolerances)),
    }


REASONING_BLOCK_START = "\n\n<assistant_reasoning>\n"
REASONING_BLOCK_END = "\n</assistant_reasoning>"
SOURCE_PROMPT_TEMPLATE_KEY = "_source_prompt_template"
LABEL_ONLY_TARGET_KEY = "_label_only_target"
DIRECT_PREDICTION_PREFIX = "Prediction:"
DIRECT_RATING_PREFIX = "Rating:"
SOFT_TARGET_KEY = "_soft_target"
CANONICAL_QWEN35_LORA_FRAGMENT = ".model.language_model.layers."
VISION_MODULE_MARKERS = ("visual", "vision_tower", "merger", "patch_embed")
DATASET_SAMPLING_MODES = ("proportional", "sqrt_balanced", "uniform_dataset")

XML_STEP_END_PATTERN = re.compile(r"</step_\d+>")
BRACKET_EVENT_PATTERN = re.compile(r"\[(USER|ASSISTANT|TOOL)\]")


def action_endpoint_character_offsets(prompt: str) -> list[int]:
    """Return deterministic action/tool-result endpoints in a rendered prompt."""
    xml_offsets = [match.end() for match in XML_STEP_END_PATTERN.finditer(prompt)]
    if xml_offsets:
        return xml_offsets
    trajectory_end = prompt.find("\n</agent_trajectory>")
    if trajectory_end < 0:
        trajectory_end = len(prompt)
    markers = [
        marker
        for marker in BRACKET_EVENT_PATTERN.finditer(prompt)
        if marker.start() < trajectory_end
    ]
    offsets = []
    for index, marker in enumerate(markers):
        if marker.group(1) not in {"ASSISTANT", "TOOL"}:
            continue
        end = markers[index + 1].start() if index + 1 < len(markers) else trajectory_end
        while end > marker.end() and prompt[end - 1].isspace():
            end -= 1
        if end > marker.end():
            offsets.append(end)
    return sorted(set(offsets))


def select_evenly_spaced_positions(positions: Sequence[int], maximum: int) -> list[int]:
    """Select at most ``maximum`` positions while retaining both endpoints."""
    unique = sorted(set(int(position) for position in positions))
    if maximum < 1:
        raise ValueError("MIL maximum instances must be positive")
    if len(unique) <= maximum:
        return unique
    if maximum == 1:
        return [unique[-1]]
    indices = {
        round(index * (len(unique) - 1) / (maximum - 1)) for index in range(maximum)
    }
    selected = [unique[index] for index in sorted(indices)]
    if len(selected) != maximum:
        raise AssertionError("even MIL selection did not retain requested count")
    return selected


def mil_token_positions(
    tokenizer: Any,
    serialized_prompt: str,
    raw_prompt: str,
    *,
    max_length: int,
    maximum_instances: int,
) -> tuple[list[int], list[int]]:
    """Map raw-prompt event endpoints into tail-truncated token positions."""
    raw_start = serialized_prompt.find(raw_prompt)
    if raw_start < 0:
        raise ValueError("serialized chat prompt does not contain raw prompt")
    encoded = tokenizer(
        serialized_prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    token_ids = list(encoded["input_ids"])
    token_offsets = list(encoded["offset_mapping"])
    valid_offsets = [
        (end, index) for index, (start, end) in enumerate(token_offsets) if end > start
    ]
    valid_ends = [end for end, _ in valid_offsets]
    cutoff = max(0, len(token_ids) - max_length)
    positions = []
    for raw_end in action_endpoint_character_offsets(raw_prompt):
        absolute_end = raw_start + raw_end
        offset_index = bisect.bisect_right(valid_ends, absolute_end) - 1
        if offset_index >= 0:
            token_index = valid_offsets[offset_index][1]
            if token_index >= cutoff:
                positions.append(token_index - cutoff)
    selected = select_evenly_spaced_positions(positions, maximum_instances)
    if not selected:
        raise ValueError("no action endpoints survive direct-prompt truncation")
    return token_ids[-max_length:], selected


def pool_mil_margins(
    margins: torch.Tensor,
    mask: torch.Tensor,
    *,
    mode: str,
    temperature: float,
    top_k: int,
) -> torch.Tensor:
    """Pool variable-size instance margins into one trajectory-level logit."""
    if margins.ndim != 2 or mask.shape != margins.shape:
        raise ValueError("MIL margins and mask must have identical rank-two shapes")
    if not bool(mask.any(dim=1).all()):
        raise ValueError("every MIL bag must contain at least one instance")
    if temperature <= 0:
        raise ValueError("MIL pooling temperature must be positive")
    if mode == "max":
        return margins.masked_fill(~mask, -torch.inf).max(dim=1).values
    if mode == "logmeanexp":
        scaled = margins.masked_fill(~mask, -torch.inf) / temperature
        counts = mask.sum(dim=1).to(margins.dtype)
        return temperature * (torch.logsumexp(scaled, dim=1) - counts.log())
    if mode == "topk_mean":
        if top_k < 1:
            raise ValueError("MIL top_k must be positive")
        pooled = []
        for row, row_mask in zip(margins, mask, strict=True):
            values = row[row_mask]
            k = min(top_k, values.numel())
            pooled.append(values.topk(k).values.mean())
        return torch.stack(pooled)
    raise ValueError(f"unknown MIL pooling mode: {mode!r}")


DATASET_LOSS_WEIGHTING_MODES = ("mean", "group_dro")


class GroupDROLoss:
    """EMA-smoothed adaptive group weighting for per-row losses."""

    def __init__(self, groups: int, *, eta: float, ema: float) -> None:
        if groups < 1:
            raise ValueError("GroupDRO requires at least one group")
        if not np.isfinite(eta) or eta <= 0:
            raise ValueError("GroupDRO eta must be finite and positive")
        if not 0 <= ema < 1:
            raise ValueError("GroupDRO EMA must lie in [0, 1)")
        self.groups = groups
        self.eta = float(eta)
        self.ema = float(ema)
        self.loss_ema: torch.Tensor | None = None
        self.seen: torch.Tensor | None = None
        self.weights: torch.Tensor | None = None

    def __call__(self, losses: torch.Tensor, dataset_ids: torch.Tensor) -> torch.Tensor:
        if losses.ndim != 1 or dataset_ids.shape != losses.shape:
            raise ValueError("GroupDRO losses and dataset ids must be row vectors")
        if not torch.isfinite(losses).all():
            raise ValueError("GroupDRO losses must be finite")
        if (dataset_ids < 0).any() or (dataset_ids >= self.groups).any():
            raise ValueError("GroupDRO dataset id outside configured range")
        device = losses.device
        if self.loss_ema is None:
            self.loss_ema = torch.zeros(self.groups, device=device, dtype=torch.float32)
            self.seen = torch.zeros(self.groups, device=device, dtype=torch.bool)
            self.weights = torch.full(
                (self.groups,), 1.0 / self.groups, device=device, dtype=torch.float32
            )
        assert self.seen is not None and self.weights is not None
        with torch.no_grad():
            for group in torch.unique(dataset_ids):
                group_index = int(group.item())
                observed = losses[dataset_ids == group].detach().float().mean()
                if self.seen[group_index]:
                    self.loss_ema[group_index] = (
                        self.ema * self.loss_ema[group_index]
                        + (1.0 - self.ema) * observed
                    )
                else:
                    self.loss_ema[group_index] = observed
                    self.seen[group_index] = True
            logits = self.eta * self.loss_ema
            logits = torch.where(self.seen, logits, torch.full_like(logits, -torch.inf))
            self.weights = torch.softmax(logits, dim=0)
        row_weights = self.weights.index_select(0, dataset_ids).to(losses.dtype)
        return (losses * row_weights).sum() / row_weights.sum().clamp_min(1e-12)

    def snapshot(self) -> dict[str, Any] | None:
        """Return JSON-compatible final EMA losses and group weights."""
        if self.loss_ema is None or self.seen is None or self.weights is None:
            return None
        return {
            "loss_ema": self.loss_ema.detach().cpu().tolist(),
            "seen": self.seen.detach().cpu().tolist(),
            "weights": self.weights.detach().cpu().tolist(),
        }


def dataset_sampling_weights(
    dataset_ids: list[int], mode: str
) -> tuple[torch.Tensor, dict[int, float]]:
    """Return per-row weights and expected dataset mass for a sampling mode."""
    if mode not in DATASET_SAMPLING_MODES:
        raise ValueError(
            f"unknown dataset sampling mode {mode!r}; "
            f"expected one of {DATASET_SAMPLING_MODES}"
        )
    if not dataset_ids:
        raise ValueError("dataset sampling requires at least one row")
    counts = Counter(int(dataset_id) for dataset_id in dataset_ids)
    exponent = {
        "proportional": 0.0,
        "sqrt_balanced": -0.5,
        "uniform_dataset": -1.0,
    }[mode]
    weights = torch.tensor(
        [counts[int(dataset_id)] ** exponent for dataset_id in dataset_ids],
        dtype=torch.double,
    )
    total_weight = float(weights.sum())
    expected_mass = {
        dataset_id: float(
            sum(
                weight
                for selected_id, weight in zip(
                    dataset_ids, weights.tolist(), strict=True
                )
                if int(selected_id) == dataset_id
            )
            / total_weight
        )
        for dataset_id in sorted(counts)
    }
    return weights, expected_mass


def validate_trainable_lora_layout(model: Any, model_loader: str) -> list[str]:
    """Reject empty or noncanonical Qwen3.5 LoRA parameter matches."""
    names = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" in name
    ]
    if not names:
        raise RuntimeError("PEFT matched zero trainable LoRA parameters")
    if model_loader == "image_text_to_text":
        noncanonical = [
            name for name in names if CANONICAL_QWEN35_LORA_FRAGMENT not in name
        ]
        if noncanonical:
            raise RuntimeError(
                "canonical Qwen3.5 training produced non-language-model LoRA "
                f"parameters, first={noncanonical[:3]}"
            )
        visual = [
            name
            for name in names
            if any(marker in name for marker in VISION_MODULE_MARKERS)
        ]
        if visual:
            raise RuntimeError(
                "canonical Qwen3.5 LoRA unexpectedly targets visual modules, "
                f"first={visual[:3]}"
            )
    return names


def parameter_counts(model: Any, finetuning_mode: str) -> dict[str, int]:
    """Validate the tuning layout and return auditable parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    lora_trainable = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" in name
    )
    auxiliary_trainable = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "decision_head" in name
    )
    if finetuning_mode == "lora":
        if not lora_trainable or trainable != lora_trainable + auxiliary_trainable:
            raise RuntimeError(
                "LoRA mode must train only LoRA and declared decision-head "
                f"parameters: trainable={trainable} lora_trainable={lora_trainable} "
                f"auxiliary_trainable={auxiliary_trainable}"
            )
    elif finetuning_mode == "full":
        if lora_trainable:
            raise RuntimeError("full fine-tuning unexpectedly contains LoRA parameters")
        if trainable != total:
            raise RuntimeError(
                "full fine-tuning must leave every model parameter trainable: "
                f"trainable={trainable} total={total}"
            )
    else:
        raise ValueError(f"unknown student.finetuning_mode={finetuning_mode!r}")
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "lora_trainable_parameters": lora_trainable,
        "auxiliary_trainable_parameters": auxiliary_trainable,
    }


def gated_delta_kernel_modules(model: torch.nn.Module) -> list[str]:
    """Return the concrete Qwen gated-delta implementations bound to the model."""
    modules = {
        kernel.__module__
        for module in model.modules()
        if (kernel := getattr(module, "chunk_gated_delta_rule", None)) is not None
    }
    return sorted(modules)


def causal_conv1d_kernel_modules(model: torch.nn.Module) -> list[str]:
    """Return the concrete Qwen causal-convolution implementations in use."""
    modules = {
        kernel.__module__ if kernel is not None else "torch_fallback"
        for module in model.modules()
        if hasattr(module, "causal_conv1d_fn")
        for kernel in [module.causal_conv1d_fn]
    }
    return sorted(modules)


def omit_redundant_attention_mask(
    attention_mask: torch.Tensor,
) -> torch.Tensor | None:
    """Let causal SDPA use its flash path when a batch contains no padding."""
    if attention_mask.ndim != 2:
        raise ValueError("attention mask must have shape [batch, sequence]")
    return None if bool(attention_mask.bool().all()) else attention_mask


def decoder_autocast(input_ids: torch.Tensor) -> Any:
    """Match Trainer's BF16 model wrapper when calling a base decoder directly."""
    return torch.autocast(
        device_type=input_ids.device.type,
        dtype=torch.bfloat16,
        enabled=input_ids.device.type == "cuda",
    )


def forward_final_token_logits(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mode: str,
) -> tuple[torch.Tensor, Any]:
    """Return per-row final-token logits without projecting padded positions."""
    last_positions = attention_mask.sum(dim=1) - 1
    if (last_positions < 0).any():
        raise ValueError("direct inputs must contain at least one attended token")
    row_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
    if mode == "full":
        outputs = model(
            input_ids=input_ids,
            attention_mask=omit_redundant_attention_mask(attention_mask),
            use_cache=False,
        )
        return outputs.logits[row_indices, last_positions], outputs
    if mode != "selected_positions":
        raise ValueError(f"unknown direct_logits_mode={mode!r}")

    selected_positions, inverse = torch.unique(
        last_positions,
        sorted=True,
        return_inverse=True,
    )
    outputs = model(
        input_ids=input_ids,
        attention_mask=omit_redundant_attention_mask(attention_mask),
        use_cache=False,
        logits_to_keep=selected_positions,
    )
    if outputs.logits.shape[1] != len(selected_positions):
        raise RuntimeError(
            "model did not honor logits_to_keep: "
            f"requested={len(selected_positions)} got={outputs.logits.shape[1]}"
        )
    return outputs.logits[row_indices, inverse], outputs


def forward_final_token_hidden(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, Any]:
    """Return final causal hidden states without passing through the LM head."""
    last_positions = attention_mask.sum(dim=1) - 1
    if (last_positions < 0).any():
        raise ValueError("direct inputs must contain at least one attended token")
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    decoder = getattr(base, "model", None)
    if decoder is None:
        raise RuntimeError("causal model exposes no decoder for decision-head mode")
    with decoder_autocast(input_ids):
        outputs = decoder(
            input_ids=input_ids,
            attention_mask=omit_redundant_attention_mask(attention_mask),
            use_cache=False,
        )
    selected_positions, inverse = torch.unique(
        last_positions,
        sorted=True,
        return_inverse=True,
    )
    selected_hidden = outputs.last_hidden_state[:, selected_positions, :]
    row_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
    return selected_hidden[row_indices, inverse], outputs


def forward_final_and_mil_binary_logits(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mil_positions: torch.Tensor,
    binary_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, Any]:
    """Score final and intermediate positions with only two LM-head rows."""
    if mil_positions.ndim != 2 or mil_positions.shape[0] != input_ids.shape[0]:
        raise ValueError("MIL positions must be [batch, instances]")
    last_positions = attention_mask.sum(dim=1) - 1
    if (last_positions < 0).any():
        raise ValueError("direct inputs must contain at least one attended token")
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    decoder = getattr(base, "model", None)
    lm_head = getattr(base, "lm_head", None)
    if decoder is None or lm_head is None or not hasattr(lm_head, "weight"):
        raise RuntimeError("causal model exposes no decoder/LM-head projection")
    with decoder_autocast(input_ids):
        outputs = decoder(
            input_ids=input_ids,
            attention_mask=omit_redundant_attention_mask(attention_mask),
            use_cache=False,
        )
    hidden = outputs.last_hidden_state
    rows = torch.arange(input_ids.shape[0], device=input_ids.device)
    final_hidden = hidden[rows, last_positions]
    mil_rows = rows[:, None].expand_as(mil_positions)
    mil_hidden = hidden[mil_rows, mil_positions]
    selected_weight = lm_head.weight.index_select(0, binary_ids)
    selected_bias = (
        None
        if getattr(lm_head, "bias", None) is None
        else lm_head.bias.index_select(0, binary_ids)
    )
    final_logits = F.linear(final_hidden, selected_weight, selected_bias)
    mil_logits = F.linear(mil_hidden, selected_weight, selected_bias)
    return final_logits, mil_logits, outputs


def selected_completion_cross_entropy(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    *,
    projection_chunk_size: int,
) -> tuple[torch.Tensor, Any, int]:
    """Compute exact causal CE without projecting unsupervised prefix positions."""
    if input_ids.shape != attention_mask.shape or labels.shape != input_ids.shape:
        raise ValueError("completion input, mask, and labels must share shape")
    if projection_chunk_size < 1:
        raise ValueError("completion projection chunk size must be positive")
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    decoder = getattr(base, "model", None)
    lm_head = getattr(base, "lm_head", None)
    if decoder is None or lm_head is None:
        raise RuntimeError("causal model exposes no decoder/LM head")
    with decoder_autocast(input_ids):
        outputs = decoder(
            input_ids=input_ids,
            attention_mask=omit_redundant_attention_mask(attention_mask),
            use_cache=False,
        )
    shifted_labels = labels[:, 1:]
    supervised = shifted_labels.ne(-100)
    target_ids = shifted_labels[supervised]
    target_hidden = outputs.last_hidden_state[:, :-1][supervised]
    if target_ids.numel() == 0:
        raise ValueError("completion batch contains no supervised tokens")
    loss_sum = target_hidden.sum() * 0.0
    for start in range(0, target_ids.numel(), projection_chunk_size):
        end = min(start + projection_chunk_size, target_ids.numel())
        logits = lm_head(target_hidden[start:end])
        loss_sum = loss_sum + F.cross_entropy(
            logits.float(),
            target_ids[start:end],
            reduction="sum",
        )
    return loss_sum / target_ids.numel(), outputs, int(target_ids.numel())


def forward_final_token_logits_and_head_inputs(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, Any]:
    """Return final logits and the exact selected inputs seen by the LM head."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    lm_head = getattr(base, "lm_head", None)
    if lm_head is None:
        raise RuntimeError("causal model exposes no LM head")
    captured: list[torch.Tensor] = []

    def capture_head_inputs(_module, arguments) -> None:
        captured.append(arguments[0])

    handle = lm_head.register_forward_pre_hook(capture_head_inputs)
    try:
        logits, outputs = forward_final_token_logits(
            model, input_ids, attention_mask, mode
        )
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"expected one LM-head invocation, captured {len(captured)}")
    last_positions = attention_mask.sum(dim=1) - 1
    _, inverse = torch.unique(last_positions, sorted=True, return_inverse=True)
    row_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
    hidden = captured[0][row_indices, inverse]
    return logits, hidden, outputs


def straight_through_decision_logits(
    head_logits: torch.Tensor, token_logits: torch.Tensor
) -> torch.Tensor:
    """Use head scores forward and LM token-row gradients for the backbone."""
    if head_logits.shape != token_logits.shape:
        raise ValueError("head and token logits must have identical shapes")
    return head_logits + (token_logits - token_logits.detach())


def binary_token_ids(tokenizer: Any) -> list[int]:
    """Return the distinct single-token ids for literal binary predictions."""
    ids = []
    for text in ("0", "1"):
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(
                f"binary target {text!r} tokenized as {encoded}, expected one token"
            )
        ids.append(int(encoded[0]))
    if len(set(ids)) != 2:
        raise ValueError(f"binary targets must have distinct token ids, got {ids}")
    return ids


def collate_eva_features(
    tokenizer: Any, features: list[dict[str, Any]]
) -> dict[str, torch.Tensor]:
    """Pad EVA inputs into the concrete dictionary required by PEFT."""
    return dict(tokenizer.pad(features, padding=True, return_tensors="pt"))


def rating_token_ids(tokenizer: Any) -> list[int]:
    """Return distinct single-token ids for literal ratings one through seven."""
    ids = []
    for text in map(str, range(1, 8)):
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(
                f"rating target {text!r} tokenized as {encoded}, expected one token"
            )
        ids.append(int(encoded[0]))
    if len(set(ids)) != 7:
        raise ValueError(f"rating targets must have distinct token ids, got {ids}")
    return ids


def order_records_for_paired_batches(
    records: list[dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    """Interleave stable within-dataset positive/negative pairs for even batches."""
    by_dataset: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for record in records:
        dataset = str(record.get("dataset", ""))
        label = int(record["label"])
        if label not in (0, 1):
            raise ValueError(f"binary label required, got {label}")
        by_dataset.setdefault(dataset, {0: [], 1: []})[label].append(record)

    def digest(*parts: Any) -> bytes:
        return hashlib.sha256(
            "\0".join(str(part) for part in (seed, *parts)).encode("utf-8")
        ).digest()

    paired: list[tuple[bytes, dict[str, Any], dict[str, Any]]] = []
    leftovers: list[tuple[bytes, dict[str, Any]]] = []
    for dataset, groups in sorted(by_dataset.items()):
        negative = sorted(
            groups[0],
            key=lambda record: digest(dataset, 0, record.get("index")),
        )
        positive = sorted(
            groups[1],
            key=lambda record: digest(dataset, 1, record.get("index")),
        )
        pair_count = min(len(negative), len(positive))
        for pair_index, (negative_record, positive_record) in enumerate(
            zip(negative[:pair_count], positive[:pair_count], strict=True)
        ):
            pair_key = digest(dataset, "pair", pair_index)
            if pair_key[0] % 2:
                negative_record, positive_record = positive_record, negative_record
            paired.append((pair_key, negative_record, positive_record))
        for record in negative[pair_count:] + positive[pair_count:]:
            leftovers.append((digest(dataset, "leftover", record.get("index")), record))

    ordered = [
        record
        for _, first, second in sorted(paired, key=lambda item: item[0])
        for record in (first, second)
    ]
    ordered.extend(record for _, record in sorted(leftovers, key=lambda item: item[0]))
    if len(ordered) != len(records) or {id(record) for record in ordered} != {
        id(record) for record in records
    }:
        raise AssertionError(
            "paired record ordering must preserve every input row once"
        )
    return ordered


def order_records_for_grouped_pair_batches(
    records: list[dict[str, Any]],
    seed: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Build same-dataset batches with equal labels whenever both are available."""
    validate_paired_batch_size(batch_size)
    half_batch = batch_size // 2
    by_dataset: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for record in records:
        dataset = str(record.get("dataset", ""))
        label = int(record["label"])
        if label not in (0, 1):
            raise ValueError(f"binary label required, got {label}")
        by_dataset.setdefault(dataset, {0: [], 1: []})[label].append(record)

    def digest(*parts: Any) -> bytes:
        return hashlib.sha256(
            "\0".join(str(part) for part in (seed, *parts)).encode("utf-8")
        ).digest()

    blocks: list[tuple[bytes, list[dict[str, Any]]]] = []
    tails: list[tuple[bytes, dict[str, Any]]] = []
    for dataset, groups in sorted(by_dataset.items()):
        negative = sorted(
            groups[0],
            key=lambda record: digest(dataset, 0, record.get("index")),
        )
        positive = sorted(
            groups[1],
            key=lambda record: digest(dataset, 1, record.get("index")),
        )
        block_index = 0
        while len(negative) >= half_batch and len(positive) >= half_batch:
            block = negative[:half_batch] + positive[:half_batch]
            del negative[:half_batch]
            del positive[:half_batch]
            block.sort(
                key=lambda record: digest(
                    dataset, "within", block_index, record.get("index")
                )
            )
            blocks.append(
                (
                    digest(dataset, "block", block_index),
                    block,
                )
            )
            block_index += 1

        # If one label has fewer than half a batch left, preserve all of those
        # examples in one final mixed block and fill it from the majority.
        if negative and positive and len(negative) + len(positive) >= batch_size:
            if len(negative) < len(positive):
                minority, majority = negative, positive
            else:
                minority, majority = positive, negative
            take_majority = batch_size - len(minority)
            block = list(minority) + majority[:take_majority]
            del majority[:take_majority]
            minority.clear()
            block.sort(
                key=lambda record: digest(
                    dataset, "within", block_index, record.get("index")
                )
            )
            blocks.append(
                (
                    digest(dataset, "block", block_index),
                    block,
                )
            )
            block_index += 1

        remainder = negative + positive
        while len(remainder) >= batch_size:
            block = remainder[:batch_size]
            del remainder[:batch_size]
            blocks.append(
                (
                    digest(dataset, "block", block_index),
                    block,
                )
            )
            block_index += 1
        tails.extend(
            (digest(dataset, "tail", record.get("index")), record)
            for record in remainder
        )

    ordered = [
        record
        for _, block in sorted(blocks, key=lambda item: item[0])
        for record in block
    ]
    ordered.extend(record for _, record in sorted(tails, key=lambda item: item[0]))
    if len(ordered) != len(records) or {id(record) for record in ordered} != {
        id(record) for record in records
    }:
        raise AssertionError("grouped pair ordering must preserve every input row once")
    return ordered


def validate_paired_batch_size(batch_size: int) -> None:
    """Require an even microbatch so consecutive label pairs stay intact."""
    if batch_size <= 0 or batch_size % 2:
        raise ValueError(
            "paired batching requires a positive even per-device batch size"
        )


def pairwise_logistic_loss(
    margins: torch.Tensor,
    labels: torch.Tensor,
    dataset_ids: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Rank positive margins above negatives from the same dataset."""
    if temperature <= 0:
        raise ValueError("pairwise temperature must be positive")
    losses = []
    for dataset_id in torch.unique(dataset_ids):
        selected = dataset_ids == dataset_id
        positives = margins[selected & (labels == 1)]
        negatives = margins[selected & (labels == 0)]
        if positives.numel() and negatives.numel():
            differences = positives[:, None] - negatives[None, :]
            losses.append(F.softplus(-differences / temperature).mean())
    if not losses:
        return margins.sum() * 0.0
    return torch.stack(losses).mean()


def soft_binary_distillation_losses(
    binary_logits: torch.Tensor,
    soft_targets: torch.Tensor,
    *,
    loss_type: str = "bce",
    target_logit_center: float = 0.0,
    target_logit_scale: float = 1.0,
    huber_delta: float = 1.0,
) -> torch.Tensor:
    """Return row-wise teacher probability or standardized-margin losses."""
    if binary_logits.ndim != 2 or binary_logits.shape[1] != 2:
        raise ValueError("binary logits must have shape (batch, 2)")
    if soft_targets.shape != binary_logits.shape[:1]:
        raise ValueError("soft targets must have shape (batch,)")
    if loss_type not in {"bce", "huber"}:
        raise ValueError(f"unsupported binary soft loss type: {loss_type}")
    if not np.isfinite(target_logit_center):
        raise ValueError("soft target logit center must be finite")
    if not np.isfinite(target_logit_scale) or target_logit_scale <= 0:
        raise ValueError("soft target logit scale must be finite and positive")
    if not np.isfinite(huber_delta) or huber_delta <= 0:
        raise ValueError("soft Huber delta must be finite and positive")

    targets = soft_targets.float()
    if not torch.isfinite(targets).all():
        raise ValueError("binary soft targets must be finite")
    if (targets <= 0).any() or (targets >= 1).any():
        raise ValueError("binary soft targets must be strictly between zero and one")

    student_margins = binary_logits[:, 1].float() - binary_logits[:, 0].float()
    identity_transform = target_logit_center == 0.0 and target_logit_scale == 1.0
    if loss_type == "bce" and identity_transform:
        return F.binary_cross_entropy_with_logits(
            student_margins, targets, reduction="none"
        )

    teacher_margins = (torch.logit(targets) - float(target_logit_center)) / float(
        target_logit_scale
    )
    if loss_type == "bce":
        transformed_targets = torch.sigmoid(teacher_margins)
        return F.binary_cross_entropy_with_logits(
            student_margins,
            transformed_targets,
            reduction="none",
        )
    return F.smooth_l1_loss(
        student_margins,
        teacher_margins,
        beta=float(huber_delta),
        reduction="none",
    )


def soft_binary_distillation_loss(
    binary_logits: torch.Tensor,
    soft_targets: torch.Tensor,
    *,
    loss_type: str = "bce",
    target_logit_center: float = 0.0,
    target_logit_scale: float = 1.0,
    huber_delta: float = 1.0,
) -> torch.Tensor:
    """Match a teacher probability or standardized margin at the binary boundary."""
    return soft_binary_distillation_losses(
        binary_logits,
        soft_targets,
        loss_type=loss_type,
        target_logit_center=target_logit_center,
        target_logit_scale=target_logit_scale,
        huber_delta=huber_delta,
    ).mean()


def soft_rating_distillation_loss(
    rating_logits: torch.Tensor,
    soft_targets: torch.Tensor,
) -> torch.Tensor:
    """Match a seven-way teacher rating distribution at the direct boundary."""
    if rating_logits.ndim != 2 or rating_logits.shape[1] != 7:
        raise ValueError("rating logits must have shape (batch, 7)")
    if soft_targets.shape != rating_logits.shape:
        raise ValueError("soft rating targets must have shape (batch, 7)")
    targets = soft_targets.float()
    if not torch.isfinite(targets).all():
        raise ValueError("soft rating targets must be finite")
    if (targets < 0).any():
        raise ValueError("soft rating targets must be non-negative")
    if not torch.allclose(
        targets.sum(dim=1),
        torch.ones(targets.shape[0], device=targets.device),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError("soft rating targets must normalize row-wise")
    return F.kl_div(
        F.log_softmax(rating_logits.float(), dim=-1),
        targets,
        reduction="batchmean",
    )


def attach_soft_teacher_targets(
    records: list[dict[str, Any]],
    artifact: Path,
) -> list[dict[str, Any]]:
    """Join a complete derived soft-target cache by dataset and row index."""
    targets: dict[tuple[str, Any], tuple[int, float]] = {}
    for line_number, line in enumerate(artifact.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        target = json.loads(line)
        key = (str(target["dataset"]), target["index"])
        if key in targets:
            raise ValueError(f"duplicate soft target at line {line_number}: {key}")
        value = float(target["soft_target"])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"soft target outside [0, 1] for {key}: {value}")
        targets[key] = (int(target["label"]), value)
    if not targets:
        raise ValueError(f"soft teacher artifact is empty: {artifact}")

    attached = []
    for record in records:
        key = (str(record.get("dataset", "")), record.get("index"))
        if key not in targets:
            raise ValueError(f"soft teacher artifact is missing training row {key}")
        label, value = targets[key]
        if label != int(record["label"]):
            raise ValueError(
                f"soft teacher label mismatch for {key}: {label} != {record['label']}"
            )
        selected = dict(record)
        selected[SOFT_TARGET_KEY] = value
        attached.append(selected)
    return attached


def has_reasoning_block(prompt: str) -> bool:
    """Return whether a rendered student prompt ends in a reasoning field."""
    start = prompt.rfind(REASONING_BLOCK_START)
    return start >= 0 and prompt.rstrip().endswith(REASONING_BLOCK_END)


def strip_reasoning_block(prompt: str) -> str:
    """Remove the final rendered reasoning field without touching prompt prose."""
    start = prompt.rfind(REASONING_BLOCK_START)
    if start < 0 or not prompt.rstrip().endswith(REASONING_BLOCK_END):
        return prompt
    return prompt[:start]


def should_drop_reasoning(
    record: dict[str, Any],
    probability: float,
    seed: int,
) -> bool:
    """Choose a stable per-row trace-dropout mask independent of row order."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            "student.reasoning_dropout_probability must be between 0 and 1"
        )
    if probability == 0.0:
        return False
    if probability == 1.0:
        return True
    key = f"{seed}\0{record.get('dataset', '')}\0{record.get('index', '')}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    return fraction < probability


def student_prompt_with_reasoning_dropout(
    record: dict[str, Any],
    probability: float,
    seed: int,
) -> tuple[str, bool]:
    """Return the cached prompt after deterministic student-side trace dropout."""
    prompt = str(record["student_prompt"])
    dropped = has_reasoning_block(prompt) and should_drop_reasoning(
        record, probability, seed
    )
    return (strip_reasoning_block(prompt), True) if dropped else (prompt, False)


def load_records(
    path: Path,
    dataset_name_contains: str | None = None,
    *,
    require_label_match: bool = True,
) -> list[dict[str, Any]]:
    records = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]
    usable = [
        record
        for record in records
        if not record.get("parse_error")
        and (not require_label_match or record.get("label_match"))
        and record.get("student_target")
        and (
            dataset_name_contains is None
            or dataset_name_contains in str(record.get("dataset", ""))
        )
    ]
    if not usable:
        raise RuntimeError(f"no usable teacher records in {path}")
    return usable


def select_records_from_manifest(
    records: list[dict[str, Any]], manifest_path: Path
) -> list[dict[str, Any]]:
    """Select an exact shared dataset/index set and verify its labels."""
    desired: dict[tuple[str, Any], int] = {}
    for line_number, line in enumerate(manifest_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        key = (str(record["dataset"]), record["index"])
        if key in desired:
            raise ValueError(f"duplicate selection key at line {line_number}: {key}")
        desired[key] = int(record["label"])
    if not desired:
        raise ValueError(f"selection manifest is empty: {manifest_path}")

    available = {
        (str(record.get("dataset", "")), record.get("index")): record
        for record in records
    }
    missing = sorted(set(desired) - set(available))
    if missing:
        raise ValueError(
            f"selection manifest has {len(missing)} unavailable rows; "
            f"first={missing[0]}"
        )
    for key, label in desired.items():
        if int(available[key]["label"]) != label:
            raise ValueError(
                f"selection manifest label mismatch for {key}: "
                f"{label} != {available[key]['label']}"
            )
    return [
        record
        for record in records
        if (str(record.get("dataset", "")), record.get("index")) in desired
    ]


def apply_label_only_manifest(
    records: list[dict[str, Any]], manifest_path: Path
) -> list[dict[str, Any]]:
    """Mark exact rows to retain only their authoritative binary target."""
    desired: dict[tuple[str, Any], int] = {}
    for line_number, line in enumerate(manifest_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        manifest_record = json.loads(line)
        key = (str(manifest_record["dataset"]), manifest_record["index"])
        if key in desired:
            raise ValueError(f"duplicate label-only key at line {line_number}: {key}")
        desired[key] = int(manifest_record["label"])
    if not desired:
        raise ValueError(f"label-only manifest is empty: {manifest_path}")

    available = {
        (str(record.get("dataset", "")), record.get("index")): record
        for record in records
    }
    missing = sorted(set(desired) - set(available))
    if missing:
        raise ValueError(
            f"label-only manifest has {len(missing)} unavailable rows; "
            f"first={missing[0]}"
        )
    for key, label in desired.items():
        if int(available[key]["label"]) != label:
            raise ValueError(
                f"label-only manifest label mismatch for {key}: "
                f"{label} != {available[key]['label']}"
            )

    marked = []
    for record in records:
        selected_record = dict(record)
        key = (str(record.get("dataset", "")), record.get("index"))
        selected_record[LABEL_ONLY_TARGET_KEY] = key in desired
        marked.append(selected_record)
    return marked


def training_warmup_steps(training_cfg: DictConfig) -> float:
    """Return the v5 warmup argument while preserving ratio-based configs."""
    configured_steps = OmegaConf.select(training_cfg, "warmup_steps", default=None)
    if configured_steps is not None:
        return float(configured_steps)
    ratio = float(training_cfg.warmup_ratio)
    if not 0.0 <= ratio < 1.0:
        raise ValueError("student.training.warmup_ratio must be in [0, 1)")
    # Transformers v5 interprets a warmup_steps float below one as a ratio.
    return ratio


def load_record_sources(
    sources: list[
        tuple[Path, str | None]
        | tuple[Path, str | None, float, int]
        | tuple[Path, str | None, float, int, str | None]
    ],
    *,
    require_label_match: bool = True,
    record_identity_field: str | None = None,
) -> list[dict[str, Any]]:
    """Load cache slices, optionally distinguishing intentional row variants."""
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, Any] | tuple[str, Any, str]] = set()
    for source in sources:
        if len(source) == 2:
            path, dataset_name_contains = source
            fraction, seed = 1.0, 0
            source_prompt_template = None
        elif len(source) == 4:
            path, dataset_name_contains, fraction, seed = source
            source_prompt_template = None
        elif len(source) == 5:
            path, dataset_name_contains, fraction, seed, source_prompt_template = source
        else:
            raise ValueError(f"invalid teacher source tuple length: {len(source)}")
        source_records = load_records(
            path,
            dataset_name_contains=dataset_name_contains,
            require_label_match=require_label_match,
        )
        source_records = select_stratified_fraction(
            source_records, float(fraction), int(seed)
        )
        print(
            f"teacher source path={path} filter={dataset_name_contains!r} "
            f"fraction={float(fraction)} seed={int(seed)} "
            f"prompt_override={source_prompt_template is not None} "
            f"selected={len(source_records)}"
        )
        for record in source_records:
            base_key = (str(record.get("dataset", "")), record.get("index"))
            if record_identity_field is None:
                key: tuple[str, Any] | tuple[str, Any, str] = base_key
            else:
                identity = record.get(record_identity_field)
                if identity is None or str(identity).strip() == "":
                    raise ValueError(
                        f"teacher record {base_key} is missing non-empty "
                        f"identity field {record_identity_field!r}"
                    )
                key = (*base_key, str(identity))
            if key in seen:
                raise ValueError(f"duplicate teacher record across sources: {key}")
            seen.add(key)
            selected_record = dict(record)
            if source_prompt_template is not None:
                selected_record[SOURCE_PROMPT_TEMPLATE_KEY] = str(
                    source_prompt_template
                )
            records.append(selected_record)
    if not records:
        raise RuntimeError("no usable records across teacher sources")
    return records


def select_stratified_fraction(
    records: list[dict[str, Any]],
    fraction: float,
    seed: int,
) -> list[dict[str, Any]]:
    """Select a stable fraction within every dataset/label stratum."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("student.train_fraction must be in (0, 1]")
    if fraction == 1.0:
        return list(records)

    strata: dict[
        tuple[str, int],
        list[tuple[bytes, tuple[str, int, Any]]],
    ] = {}
    for record in records:
        key = (
            str(record.get("dataset", "")),
            int(record["label"]),
            record.get("index"),
        )
        stratum = key[:2]
        digest = hashlib.sha256(
            f"{seed}\0{key[0]}\0{key[1]}\0{key[2]}".encode()
        ).digest()
        strata.setdefault(stratum, []).append((digest, key))

    selected: set[tuple[str, int, Any]] = set()
    for candidates in strata.values():
        count = max(1, int(len(candidates) * fraction + 0.5))
        selected.update(key for _, key in sorted(candidates)[:count])
    return [
        record
        for record in records
        if (
            str(record.get("dataset", "")),
            int(record["label"]),
            record.get("index"),
        )
        in selected
    ]


def select_rating_uncertainty_fraction(
    records: list[dict[str, Any]],
    fraction: float,
    seed: int,
    *,
    midpoint: int = 4,
) -> list[dict[str, Any]]:
    """Select the ratings nearest the neutral midpoint within each stratum."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("rating_uncertainty_fraction must be in (0, 1]")
    if not 1 <= midpoint <= 7:
        raise ValueError("rating uncertainty midpoint must be between 1 and 7")
    if fraction == 1.0:
        return list(records)

    strata: dict[
        tuple[str, int],
        list[tuple[int, bytes, tuple[str, int, Any]]],
    ] = {}
    for record in records:
        rating = record.get("rating")
        if not isinstance(rating, int) or not 1 <= rating <= 7:
            raise ValueError(
                "rating uncertainty selection requires integer ratings 1--7"
            )
        key = (
            str(record.get("dataset", "")),
            int(record["label"]),
            record.get("index"),
        )
        digest = hashlib.sha256(
            f"{seed}\0{key[0]}\0{key[1]}\0{key[2]}".encode()
        ).digest()
        strata.setdefault(key[:2], []).append((abs(rating - midpoint), digest, key))

    selected: set[tuple[str, int, Any]] = set()
    for candidates in strata.values():
        count = max(1, int(len(candidates) * fraction + 0.5))
        selected.update(key for _, _, key in sorted(candidates)[:count])
    return [
        record
        for record in records
        if (
            str(record.get("dataset", "")),
            int(record["label"]),
            record.get("index"),
        )
        in selected
    ]


def select_rating_uncertainty_with_certain_anchors(
    records: list[dict[str, Any]],
    fraction_each: float,
    seed: int,
    *,
    midpoint: int = 4,
) -> list[dict[str, Any]]:
    """Select equal midpoint-near and extreme sets within every stratum."""
    if not 0.0 < fraction_each <= 0.5:
        raise ValueError("rating anchor fraction must be in (0, 0.5]")
    if not 1 <= midpoint <= 7:
        raise ValueError("rating uncertainty midpoint must be between 1 and 7")

    strata: dict[
        tuple[str, int],
        list[tuple[int, bytes, tuple[str, int, Any]]],
    ] = {}
    for record in records:
        rating = record.get("rating")
        if not isinstance(rating, int) or not 1 <= rating <= 7:
            raise ValueError("rating anchor selection requires integer ratings 1--7")
        key = (
            str(record.get("dataset", "")),
            int(record["label"]),
            record.get("index"),
        )
        digest = hashlib.sha256(
            f"{seed}\0{key[0]}\0{key[1]}\0{key[2]}".encode()
        ).digest()
        strata.setdefault(key[:2], []).append((abs(rating - midpoint), digest, key))

    selected: set[tuple[str, int, Any]] = set()
    for stratum, candidates in strata.items():
        count = max(1, int(len(candidates) * fraction_each + 0.5))
        if count * 2 > len(candidates):
            raise ValueError(
                f"rating anchor sets overlap in stratum={stratum!r}: "
                f"2 * {count} > {len(candidates)}"
            )
        uncertain = sorted(candidates)[:count]
        certain = sorted(candidates, key=lambda item: (-item[0], item[1]))[:count]
        selected.update(key for _, _, key in uncertain)
        selected.update(key for _, _, key in certain)
    return [
        record
        for record in records
        if (
            str(record.get("dataset", "")),
            int(record["label"]),
            record.get("index"),
        )
        in selected
    ]


def tokenize_record(
    record: dict[str, Any],
    tokenizer: Any,
    max_length: int,
    *,
    completion_max_length: int | None = None,
    prompt_template: str | None = None,
    prompt_template_without_reasoning: str | None = None,
    target_mode: str = "teacher",
    reasoning_dropout_probability: float = 0.0,
    reasoning_dropout_seed: int = 0,
    include_direct_target: bool = False,
    include_completion_target: bool = True,
    direct_target_prefix: str = DIRECT_PREDICTION_PREFIX,
    dataset_id: int | None = None,
    include_mil_target: bool = False,
    mil_max_instances: int = 8,
) -> dict[str, Any]:
    effective_completion_max_length = (
        max_length if completion_max_length is None else completion_max_length
    )
    if not 1 <= effective_completion_max_length <= max_length:
        raise ValueError("completion_max_length must be in [1, max_length]")
    raw_prompt, _ = student_prompt_with_reasoning_dropout(
        record,
        reasoning_dropout_probability,
        reasoning_dropout_seed,
    )
    source_prompt_template = record.get(SOURCE_PROMPT_TEMPLATE_KEY)
    effective_prompt_template = (
        str(source_prompt_template)
        if source_prompt_template is not None
        else prompt_template
    )
    if effective_prompt_template is not None:
        _, separator, evidence = raw_prompt.partition("<context>")
        if not separator:
            raise ValueError(
                f"student prompt is missing <context> for index={record['index']}"
            )
        selected_template = effective_prompt_template
        if (
            source_prompt_template is None
            and not has_reasoning_block(raw_prompt)
            and prompt_template_without_reasoning
        ):
            selected_template = prompt_template_without_reasoning
        raw_prompt = f"{selected_template}\n\n<context>{evidence}"
    direct_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": raw_prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    tokenized: dict[str, Any] = {}
    if include_completion_target:
        completion_prompt = str(record.get("completion_prompt", raw_prompt))
        completion_messages = []
        if completion_system_prompt := record.get("completion_system_prompt"):
            completion_messages.append(
                {"role": "system", "content": str(completion_system_prompt)}
            )
        completion_messages.append({"role": "user", "content": completion_prompt})
        prompt = tokenizer.apply_chat_template(
            completion_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        effective_target_mode = (
            "prediction_only" if record.get(LABEL_ONLY_TARGET_KEY) else target_mode
        )
        if effective_target_mode == "teacher":
            target = record["student_target"]
        elif effective_target_mode == "prediction_only":
            target = f"Prediction:{int(record['label'])}"
        else:
            raise ValueError(f"unknown student.target_mode={effective_target_mode!r}")
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        target_ids = tokenizer.encode(
            target + (tokenizer.eos_token or ""),
            add_special_tokens=False,
        )
        if len(target_ids) >= effective_completion_max_length:
            raise ValueError(
                "target alone exceeds student.completion_max_length for "
                f"index={record['index']}"
            )
        prompt_ids = prompt_ids[-(effective_completion_max_length - len(target_ids)) :]
        tokenized.update(
            input_ids=prompt_ids + target_ids,
            labels=[-100] * len(prompt_ids) + target_ids,
        )
    elif not include_direct_target:
        raise ValueError("at least one tokenized training target must be materialized")
    if include_direct_target:
        if dataset_id is None:
            raise ValueError("dataset_id is required for direct-target training")
        serialized_direct = direct_prompt + direct_target_prefix
        if include_mil_target:
            direct_ids, mil_positions = mil_token_positions(
                tokenizer,
                serialized_direct,
                raw_prompt,
                max_length=max_length,
                maximum_instances=mil_max_instances,
            )
            tokenized["mil_positions"] = mil_positions
        else:
            direct_ids = tokenizer.encode(
                serialized_direct,
                add_special_tokens=False,
            )[-max_length:]
        if not direct_ids:
            raise ValueError(f"empty direct prompt for index={record['index']}")
        tokenized.update(
            {
                "direct_input_ids": direct_ids,
                "binary_label": int(record["label"]),
                "dataset_id": int(dataset_id),
            }
        )
        if SOFT_TARGET_KEY in record:
            tokenized["soft_target"] = float(record[SOFT_TARGET_KEY])
        if "soft_rating_probs" in record:
            probabilities = record["soft_rating_probs"]
            tokenized["soft_rating_probs"] = [
                float(probabilities[str(rating)]) for rating in range(1, 8)
            ]
    return tokenized


@hydra.main(
    version_base=None,
    config_path=".",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    from datasets import Dataset
    from peft import (
        EvaConfig,
        LoraConfig,
        PeftModel,
        get_peft_model,
        initialize_lora_eva_weights,
        prepare_model_for_kbit_training,
    )
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )
    from transformers.utils.import_utils import (
        is_causal_conv1d_available,
        is_flash_linear_attention_available,
    )

    class OptimizerStepTimer(TrainerCallback):
        """Measure synchronized wall time for complete optimizer steps."""

        def __init__(self) -> None:
            self.started_at: float | None = None
            self.durations: list[float] = []

        @staticmethod
        def synchronize() -> None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        def on_step_begin(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> None:
            self.synchronize()
            self.started_at = time.perf_counter()

        def on_step_end(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> None:
            if self.started_at is None:
                return
            self.synchronize()
            self.durations.append(time.perf_counter() - self.started_at)
            self.started_at = None

        def summary(self, warmup_steps: int = 2) -> dict[str, Any]:
            steady = self.durations[min(warmup_steps, len(self.durations)) :]
            return {
                "durations_seconds": self.durations,
                "recorded_steps": len(self.durations),
                "warmup_steps_excluded": min(warmup_steps, len(self.durations)),
                "steady_steps": len(steady),
                "steady_mean_seconds": (float(np.mean(steady)) if steady else None),
                "steady_median_seconds": (float(np.median(steady)) if steady else None),
                "steady_p90_seconds": (
                    float(np.percentile(steady, 90)) if steady else None
                ),
            }

    direct_loss_weight = float(
        OmegaConf.select(cfg, "student.training.direct_loss_weight", default=0.0)
    )
    completion_loss_weight = float(
        OmegaConf.select(cfg, "student.training.completion_loss_weight", default=1.0)
    )
    completion_logits_mode = str(
        OmegaConf.select(cfg, "student.training.completion_logits_mode", default="full")
    )
    completion_projection_chunk_size = int(
        OmegaConf.select(
            cfg,
            "student.training.completion_projection_chunk_size",
            default=128,
        )
    )
    sequential_objective_backward = bool(
        OmegaConf.select(
            cfg,
            "student.training.sequential_objective_backward",
            default=False,
        )
    )
    pairwise_loss_weight = float(
        OmegaConf.select(cfg, "student.training.pairwise_loss_weight", default=0.0)
    )
    soft_loss_weight = float(
        OmegaConf.select(cfg, "student.training.soft_loss_weight", default=0.0)
    )
    mil_loss_weight = float(
        OmegaConf.select(cfg, "student.training.mil_loss_weight", default=0.0)
    )
    mil_pooling = str(
        OmegaConf.select(cfg, "student.training.mil_pooling", default="logmeanexp")
    )
    mil_temperature = float(
        OmegaConf.select(cfg, "student.training.mil_temperature", default=1.0)
    )
    mil_top_k = int(OmegaConf.select(cfg, "student.training.mil_top_k", default=3))
    mil_max_instances = int(
        OmegaConf.select(cfg, "student.training.mil_max_instances", default=8)
    )
    soft_loss_type = str(
        OmegaConf.select(cfg, "student.training.soft_loss_type", default="bce")
    )
    direct_logits_mode = str(
        OmegaConf.select(
            cfg,
            "student.training.direct_logits_mode",
            default="full",
        )
    )
    if direct_logits_mode not in {"full", "selected_positions"}:
        raise ValueError(
            "student.training.direct_logits_mode must be full or selected_positions"
        )
    decision_head_mode = str(
        OmegaConf.select(
            cfg, "student.training.decision_head_mode", default="token_logits"
        )
    )
    if decision_head_mode not in {"token_logits", "binary_head"}:
        raise ValueError(
            "student.training.decision_head_mode must be token_logits or binary_head"
        )
    decision_head_init = str(
        OmegaConf.select(cfg, "student.training.decision_head_init", default="random")
    )
    if decision_head_init not in {"random", "token_rows"}:
        raise ValueError(
            "student.training.decision_head_init must be random or token_rows"
        )
    if decision_head_mode == "token_logits" and decision_head_init != "random":
        raise ValueError("token-row initialization requires binary_head mode")
    soft_target_logit_center = float(
        OmegaConf.select(cfg, "student.training.soft_target_logit_center", default=0.0)
    )
    soft_target_logit_scale = float(
        OmegaConf.select(cfg, "student.training.soft_target_logit_scale", default=1.0)
    )
    soft_huber_delta = float(
        OmegaConf.select(cfg, "student.training.soft_huber_delta", default=1.0)
    )
    ordinal_soft_loss_weight = float(
        OmegaConf.select(cfg, "student.training.ordinal_soft_loss_weight", default=0.0)
    )
    pairwise_temperature = float(
        OmegaConf.select(cfg, "student.training.pairwise_temperature", default=1.0)
    )
    paired_batching = bool(
        OmegaConf.select(cfg, "student.training.paired_batching", default=False)
    )
    paired_batching_mode = str(
        OmegaConf.select(
            cfg, "student.training.paired_batching_mode", default="interleaved"
        )
    )
    dataset_sampling = str(
        OmegaConf.select(
            cfg,
            "student.training.dataset_sampling",
            default="proportional",
        )
    )
    if dataset_sampling not in DATASET_SAMPLING_MODES:
        raise ValueError(
            "student.training.dataset_sampling must be one of: "
            + ", ".join(DATASET_SAMPLING_MODES)
        )
    train_sampling_strategy = str(
        OmegaConf.select(
            cfg,
            "student.training.train_sampling_strategy",
            default="random",
        )
    )
    if train_sampling_strategy not in {
        "random",
        "group_by_length",
        "sequential",
    }:
        raise ValueError(
            "student.training.train_sampling_strategy must be random, "
            "group_by_length, or sequential"
        )
    dataset_loss_weighting = str(
        OmegaConf.select(
            cfg,
            "student.training.dataset_loss_weighting",
            default="mean",
        )
    )
    if dataset_loss_weighting not in DATASET_LOSS_WEIGHTING_MODES:
        raise ValueError(
            "student.training.dataset_loss_weighting must be one of: "
            + ", ".join(DATASET_LOSS_WEIGHTING_MODES)
        )
    group_dro_eta = float(
        OmegaConf.select(cfg, "student.training.group_dro_eta", default=2.0)
    )
    group_dro_ema = float(
        OmegaConf.select(cfg, "student.training.group_dro_ema", default=0.9)
    )
    if any(
        weight < 0
        for weight in (
            completion_loss_weight,
            direct_loss_weight,
            pairwise_loss_weight,
            soft_loss_weight,
            mil_loss_weight,
            ordinal_soft_loss_weight,
        )
    ):
        raise ValueError("auxiliary loss weights must be non-negative")
    if not any(
        (
            completion_loss_weight,
            direct_loss_weight,
            pairwise_loss_weight,
            soft_loss_weight,
            mil_loss_weight,
            ordinal_soft_loss_weight,
        )
    ):
        raise ValueError("at least one student loss weight must be positive")
    if ordinal_soft_loss_weight and (
        direct_loss_weight
        or pairwise_loss_weight
        or soft_loss_weight
        or mil_loss_weight
    ):
        raise ValueError(
            "ordinal soft loss cannot be combined with binary auxiliary losses"
        )
    if pairwise_temperature <= 0:
        raise ValueError("student.training.pairwise_temperature must be positive")
    if soft_loss_type not in {"bce", "huber"}:
        raise ValueError("student.training.soft_loss_type must be one of: bce, huber")
    if mil_pooling not in {"max", "logmeanexp", "topk_mean"}:
        raise ValueError("student.training.mil_pooling is invalid")
    if completion_logits_mode not in {"full", "selected_positions"}:
        raise ValueError("completion logits mode must be full or selected_positions")
    if completion_projection_chunk_size < 1:
        raise ValueError("completion projection chunk size must be positive")
    if mil_temperature <= 0 or mil_top_k < 1 or mil_max_instances < 1:
        raise ValueError(
            "MIL temperature, top-k, and maximum instances must be positive"
        )
    if mil_loss_weight and not soft_loss_weight:
        raise ValueError("MIL currently requires the Kimi soft-target objective")
    if mil_loss_weight and decision_head_mode != "token_logits":
        raise ValueError("MIL currently requires token-logit decision mode")
    if not np.isfinite(soft_target_logit_center):
        raise ValueError("student.training.soft_target_logit_center must be finite")
    if not np.isfinite(soft_target_logit_scale) or soft_target_logit_scale <= 0:
        raise ValueError(
            "student.training.soft_target_logit_scale must be finite and positive"
        )
    if not np.isfinite(soft_huber_delta) or soft_huber_delta <= 0:
        raise ValueError(
            "student.training.soft_huber_delta must be finite and positive"
        )
    if pairwise_loss_weight and not paired_batching:
        raise ValueError("pairwise loss requires student.training.paired_batching=true")
    if paired_batching and dataset_sampling != "proportional":
        raise ValueError("paired batching is incompatible with dataset reweighting")
    if dataset_loss_weighting == "group_dro" and (
        not soft_loss_weight
        or completion_loss_weight
        or direct_loss_weight
        or pairwise_loss_weight
        or mil_loss_weight
        or ordinal_soft_loss_weight
    ):
        raise ValueError("GroupDRO currently requires the binary soft loss alone")
    if paired_batching_mode not in {"interleaved", "same_dataset"}:
        raise ValueError(
            "student.training.paired_batching_mode must be interleaved or same_dataset"
        )
    uses_direct_forward = bool(
        direct_loss_weight
        or pairwise_loss_weight
        or soft_loss_weight
        or mil_loss_weight
        or ordinal_soft_loss_weight
    )
    if sequential_objective_backward and not (
        completion_loss_weight and uses_direct_forward
    ):
        raise ValueError(
            "sequential objective backward requires completion and direct losses"
        )
    group_dro_loss: GroupDROLoss | None = None

    class AuxiliarySFTTrainer(Trainer):
        """Completion SFT with optional direct-label and within-dataset rank losses."""

        def _get_train_sampler(self, train_dataset=None):
            if paired_batching:
                selected_dataset = (
                    self.train_dataset if train_dataset is None else train_dataset
                )
                return SequentialSampler(selected_dataset)
            if dataset_sampling != "proportional":
                selected_dataset = (
                    self.train_dataset if train_dataset is None else train_dataset
                )
                weights, _ = dataset_sampling_weights(
                    [int(value) for value in selected_dataset["dataset_id"]],
                    dataset_sampling,
                )
                generator = torch.Generator()
                generator.manual_seed(int(cfg.seed))
                return WeightedRandomSampler(
                    weights,
                    num_samples=len(selected_dataset),
                    replacement=True,
                    generator=generator,
                )
            return super()._get_train_sampler(train_dataset)

        def compute_loss(
            self,
            model,
            inputs,
            return_outputs=False,
            num_items_in_batch=None,
        ):
            direct_input_ids = inputs.pop("direct_input_ids", None)
            direct_attention_mask = inputs.pop("direct_attention_mask", None)
            binary_labels = inputs.pop("binary_labels", None)
            dataset_ids = inputs.pop("dataset_ids", None)
            soft_targets = inputs.pop("soft_targets", None)
            soft_rating_targets = inputs.pop("soft_rating_targets", None)
            mil_positions = inputs.pop("mil_positions", None)
            mil_position_mask = inputs.pop("mil_position_mask", None)
            outputs = None
            loss = None
            if completion_loss_weight:
                if completion_logits_mode == "selected_positions":
                    completion_loss, outputs, _ = selected_completion_cross_entropy(
                        model,
                        inputs["input_ids"],
                        inputs["attention_mask"],
                        inputs["labels"],
                        projection_chunk_size=completion_projection_chunk_size,
                    )
                    loss = completion_loss_weight * completion_loss
                else:
                    outputs = model(**inputs)
                    completion_loss = outputs.loss
                completion_term = completion_loss_weight * completion_loss
                if sequential_objective_backward:
                    self.accelerator.backward(
                        completion_term / self.current_gradient_accumulation_steps
                    )
                    loss = completion_term.detach()
                    del outputs
                    outputs = None
                else:
                    loss = completion_term
            if uses_direct_forward:
                if any(
                    value is None
                    for value in (
                        direct_input_ids,
                        direct_attention_mask,
                        binary_labels,
                        dataset_ids,
                    )
                ):
                    raise ValueError(
                        "direct-target fields are missing from training batch"
                    )
                if decision_head_mode == "binary_head":
                    next_logits, hidden, direct_outputs = (
                        forward_final_token_logits_and_head_inputs(
                            model,
                            direct_input_ids,
                            direct_attention_mask,
                            direct_logits_mode,
                        )
                    )
                    base = model.get_base_model()
                    label_ids = torch.tensor(
                        self.direct_target_ids,
                        device=next_logits.device,
                        dtype=torch.long,
                    )
                    token_logits = next_logits.index_select(-1, label_ids)
                    head_logits = base.decision_head(hidden.detach().float())
                    direct_logits = straight_through_decision_logits(
                        head_logits, token_logits
                    )
                else:
                    label_ids = torch.tensor(
                        self.direct_target_ids,
                        device=direct_input_ids.device,
                        dtype=torch.long,
                    )
                    mil_logits = None
                    if mil_loss_weight:
                        if mil_positions is None or mil_position_mask is None:
                            raise ValueError("MIL position fields are missing")
                        direct_logits, mil_logits, direct_outputs = (
                            forward_final_and_mil_binary_logits(
                                model,
                                direct_input_ids,
                                direct_attention_mask,
                                mil_positions,
                                label_ids,
                            )
                        )
                    else:
                        next_logits, direct_outputs = forward_final_token_logits(
                            model,
                            direct_input_ids,
                            direct_attention_mask,
                            direct_logits_mode,
                        )
                        direct_logits = next_logits.index_select(-1, label_ids)
                if loss is None:
                    loss = direct_logits.sum() * 0.0
                if direct_loss_weight:
                    loss = loss + direct_loss_weight * F.cross_entropy(
                        direct_logits.float(), binary_labels
                    )
                if pairwise_loss_weight:
                    margins = direct_logits[:, 1].float() - direct_logits[:, 0].float()
                    loss = loss + pairwise_loss_weight * pairwise_logistic_loss(
                        margins,
                        binary_labels,
                        dataset_ids,
                        pairwise_temperature,
                    )
                if soft_loss_weight:
                    if soft_targets is None:
                        raise ValueError("soft targets are missing from training batch")
                    soft_losses = soft_binary_distillation_losses(
                        direct_logits,
                        soft_targets,
                        loss_type=soft_loss_type,
                        target_logit_center=soft_target_logit_center,
                        target_logit_scale=soft_target_logit_scale,
                        huber_delta=soft_huber_delta,
                    )
                    soft_loss = (
                        group_dro_loss(soft_losses, dataset_ids)
                        if group_dro_loss is not None
                        else soft_losses.mean()
                    )
                    loss = loss + soft_loss_weight * soft_loss
                if mil_loss_weight:
                    if soft_targets is None or mil_logits is None:
                        raise ValueError(
                            "MIL requires soft targets and instance logits"
                        )
                    margins = mil_logits[..., 1].float() - mil_logits[..., 0].float()
                    bag_logits = pool_mil_margins(
                        margins,
                        mil_position_mask,
                        mode=mil_pooling,
                        temperature=mil_temperature,
                        top_k=mil_top_k,
                    )
                    loss = loss + mil_loss_weight * F.binary_cross_entropy_with_logits(
                        bag_logits,
                        soft_targets.float(),
                    )
                if ordinal_soft_loss_weight:
                    if soft_rating_targets is None:
                        raise ValueError(
                            "soft rating targets are missing from training batch"
                        )
                    loss = (
                        loss
                        + ordinal_soft_loss_weight
                        * soft_rating_distillation_loss(
                            direct_logits,
                            soft_rating_targets,
                        )
                    )
            if loss is None:
                raise AssertionError("configured losses produced no training loss")
            if return_outputs:
                return loss, outputs if outputs is not None else direct_outputs
            return loss

    class MuonAuxiliarySFTTrainer(AuxiliarySFTTrainer):
        """Auxiliary SFT trainer using Muon for 2D LoRA matrices."""

        def create_optimizer(self) -> torch.optim.Optimizer:
            if self.optimizer is None:
                self.optimizer = MuonAdamW(
                    muon_adamw_param_groups(self.model, float(self.args.weight_decay)),
                    lr=float(self.args.learning_rate),
                    muon_momentum=float(cfg.student.training.muon_momentum),
                    muon_nesterov=bool(cfg.student.training.muon_nesterov),
                    muon_ns_steps=int(cfg.student.training.muon_ns_steps),
                    muon_adjust_lr_fn=str(cfg.student.training.muon_adjust_lr_fn),
                )
            return self.optimizer

    root = Path(get_original_cwd()).resolve()
    random.seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))

    teacher_sources = OmegaConf.select(cfg, "student.teacher_sources", default=None)
    if teacher_sources is None:
        artifact = Path(str(cfg.teacher.artifact))
        if not artifact.is_absolute():
            artifact = root / artifact
        dataset_name_contains = (
            None
            if cfg.student.dataset_name_contains is None
            else str(cfg.student.dataset_name_contains)
        )
        sources = [(artifact, dataset_name_contains, 1.0, int(cfg.seed))]
    else:
        sources = []
        for source in teacher_sources:
            artifact = Path(str(source.artifact))
            if not artifact.is_absolute():
                artifact = root / artifact
            source_filter = OmegaConf.select(
                source, "dataset_name_contains", default=None
            )
            source_fraction = float(
                OmegaConf.select(source, "train_fraction", default=1.0)
            )
            source_seed = int(
                OmegaConf.select(source, "train_fraction_seed", default=cfg.seed)
            )
            source_prompt = OmegaConf.select(source, "prompt", default=None)
            sources.append(
                (
                    artifact,
                    None if source_filter is None else str(source_filter),
                    source_fraction,
                    source_seed,
                    None if source_prompt is None else str(source_prompt),
                )
            )
        dataset_name_contains = "multi-source"
    require_teacher_label_match = bool(
        OmegaConf.select(cfg, "student.require_teacher_label_match", default=True)
    )
    record_identity_field_value = OmegaConf.select(
        cfg, "student.record_identity_field", default=None
    )
    record_identity_field = (
        None
        if record_identity_field_value is None
        else str(record_identity_field_value)
    )
    records = load_record_sources(
        sources,
        require_label_match=require_teacher_label_match,
        record_identity_field=record_identity_field,
    )
    train_fraction = float(OmegaConf.select(cfg, "student.train_fraction", default=1.0))
    train_fraction_seed = int(
        OmegaConf.select(cfg, "student.train_fraction_seed", default=cfg.seed)
    )
    records_before_fraction = len(records)
    selection_manifest = OmegaConf.select(
        cfg, "student.selection_manifest", default=None
    )
    rating_uncertainty_fraction = OmegaConf.select(
        cfg, "student.rating_uncertainty_fraction", default=None
    )
    if selection_manifest is not None:
        if train_fraction != 1.0 or rating_uncertainty_fraction is not None:
            raise ValueError(
                "selection_manifest requires train_fraction=1.0 and no "
                "rating_uncertainty_fraction"
            )
        manifest_path = Path(str(selection_manifest))
        if not manifest_path.is_absolute():
            manifest_path = root / manifest_path
        records = select_records_from_manifest(records, manifest_path)
        selection_mode = "fixed_manifest"
    elif rating_uncertainty_fraction is None:
        records = select_stratified_fraction(
            records,
            train_fraction,
            train_fraction_seed,
        )
        selection_mode = "random_stratified"
    else:
        if train_fraction != 1.0:
            raise ValueError(
                "set student.train_fraction=1.0 when using "
                "student.rating_uncertainty_fraction"
            )
        rating_uncertainty_seed = int(
            OmegaConf.select(
                cfg,
                "student.rating_uncertainty_seed",
                default=train_fraction_seed,
            )
        )
        rating_balance_certain = bool(
            OmegaConf.select(
                cfg,
                "student.rating_balance_certain",
                default=False,
            )
        )
        if rating_balance_certain:
            records = select_rating_uncertainty_with_certain_anchors(
                records,
                float(rating_uncertainty_fraction),
                rating_uncertainty_seed,
            )
            selection_mode = "rating_uncertainty_certain_balanced"
        else:
            records = select_rating_uncertainty_fraction(
                records,
                float(rating_uncertainty_fraction),
                rating_uncertainty_seed,
            )
            selection_mode = "rating_uncertainty_stratified"
    if cfg.student.train_limit is not None:
        records = records[: int(cfg.student.train_limit)]
    label_only_manifest = OmegaConf.select(
        cfg, "student.label_only_manifest", default=None
    )
    if label_only_manifest is not None:
        label_only_path = Path(str(label_only_manifest))
        if not label_only_path.is_absolute():
            label_only_path = root / label_only_path
        records = apply_label_only_manifest(records, label_only_path)
    soft_teacher_artifact = OmegaConf.select(
        cfg, "student.soft_teacher_artifact", default=None
    )
    if soft_loss_weight:
        if soft_teacher_artifact is None:
            raise ValueError(
                "student.soft_teacher_artifact is required when soft loss is enabled"
            )
        soft_teacher_path = Path(str(soft_teacher_artifact))
        if not soft_teacher_path.is_absolute():
            soft_teacher_path = root / soft_teacher_path
        records = attach_soft_teacher_targets(records, soft_teacher_path)
    elif soft_teacher_artifact is not None:
        raise ValueError(
            "student.soft_teacher_artifact requires a positive soft_loss_weight"
        )
    label_only_rows = sum(bool(record.get(LABEL_ONLY_TARGET_KEY)) for record in records)
    if paired_batching:
        micro_batch_size = int(cfg.student.training.per_device_train_batch_size)
        validate_paired_batch_size(micro_batch_size)
        if paired_batching_mode == "same_dataset":
            records = order_records_for_grouped_pair_batches(
                records,
                int(cfg.seed),
                micro_batch_size,
            )
        else:
            records = order_records_for_paired_batches(records, int(cfg.seed))
    dataset_id_by_name = {
        name: dataset_id
        for dataset_id, name in enumerate(
            sorted({str(record.get("dataset", "")) for record in records})
        )
    }
    if dataset_loss_weighting == "group_dro":
        group_dro_loss = GroupDROLoss(
            len(dataset_id_by_name), eta=group_dro_eta, ema=group_dro_ema
        )
    model_revision_value = OmegaConf.select(cfg, "student.model_revision", default=None)
    model_revision = None if model_revision_value is None else str(model_revision_value)
    revision_kwargs = {} if model_revision is None else {"revision": model_revision}
    tokenizer = AutoTokenizer.from_pretrained(str(cfg.student.model), **revision_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    direct_target_ids = None
    direct_target_prefix = DIRECT_PREDICTION_PREFIX
    if ordinal_soft_loss_weight:
        direct_target_ids = rating_token_ids(tokenizer)
        direct_target_prefix = DIRECT_RATING_PREFIX
    elif uses_direct_forward:
        direct_target_ids = binary_token_ids(tokenizer)
    reasoning_dropout_probability = float(
        OmegaConf.select(cfg, "student.reasoning_dropout_probability", default=0.0)
    )
    reasoning_dropout_seed = int(
        OmegaConf.select(cfg, "student.reasoning_dropout_seed", default=cfg.seed)
    )
    reasoning_rows = sum(
        has_reasoning_block(str(record["student_prompt"])) for record in records
    )
    reasoning_rows_dropped = sum(
        student_prompt_with_reasoning_dropout(
            record,
            reasoning_dropout_probability,
            reasoning_dropout_seed,
        )[1]
        for record in records
    )
    max_length = int(cfg.student.max_length)
    completion_max_length = int(
        OmegaConf.select(
            cfg,
            "student.completion_max_length",
            default=max_length,
        )
    )
    if not 1 <= completion_max_length <= max_length:
        raise ValueError(
            "student.completion_max_length must be in [1, student.max_length]"
        )
    tokenized = [
        tokenize_record(
            record,
            tokenizer,
            max_length,
            completion_max_length=completion_max_length,
            prompt_template=(
                str(cfg.student.prompt)
                if (
                    str(cfg.student.target_mode) == "prediction_only"
                    or bool(
                        OmegaConf.select(
                            cfg, "student.override_cached_prompt", default=False
                        )
                    )
                )
                else None
            ),
            prompt_template_without_reasoning=(
                str(cfg.student.prompt_without_reasoning)
                if (
                    OmegaConf.select(
                        cfg, "student.prompt_without_reasoning", default=None
                    )
                    is not None
                    and (
                        str(cfg.student.target_mode) == "prediction_only"
                        or bool(
                            OmegaConf.select(
                                cfg, "student.override_cached_prompt", default=False
                            )
                        )
                    )
                )
                else None
            ),
            target_mode=str(cfg.student.target_mode),
            reasoning_dropout_probability=reasoning_dropout_probability,
            reasoning_dropout_seed=reasoning_dropout_seed,
            include_direct_target=uses_direct_forward,
            include_completion_target=bool(completion_loss_weight),
            direct_target_prefix=direct_target_prefix,
            dataset_id=dataset_id_by_name[str(record.get("dataset", ""))],
            include_mil_target=bool(mil_loss_weight),
            mil_max_instances=mil_max_instances,
        )
        for record in records
    ]
    for feature in tokenized:
        length_source = feature.get("direct_input_ids")
        if length_source is None:
            length_source = feature["input_ids"]
        feature["length"] = len(length_source)
    dataset = Dataset.from_list(tokenized)
    sampling_weights, expected_dataset_mass_by_id = dataset_sampling_weights(
        [dataset_id_by_name[str(record.get("dataset", ""))] for record in records],
        dataset_sampling,
    )
    del sampling_weights
    dataset_name_by_id = {
        dataset_id: name for name, dataset_id in dataset_id_by_name.items()
    }
    expected_dataset_mass = {
        dataset_name_by_id[dataset_id]: mass
        for dataset_id, mass in expected_dataset_mass_by_id.items()
    }
    print(
        f"training on {len(dataset)} parsed, label-consistent teacher targets "
        f"records_before_fraction={records_before_fraction} "
        f"train_fraction={train_fraction} "
        f"train_fraction_seed={train_fraction_seed} "
        f"selection_mode={selection_mode} "
        f"rating_uncertainty_fraction={rating_uncertainty_fraction} "
        f"selection_manifest={selection_manifest} "
        f"label_only_manifest={label_only_manifest} "
        f"label_only_rows={label_only_rows} "
        f"record_identity_field={record_identity_field!r} "
        f"require_teacher_label_match={require_teacher_label_match} "
        f"dataset_name_contains={dataset_name_contains!r} "
        f"reasoning_rows={reasoning_rows} "
        f"reasoning_rows_dropped={reasoning_rows_dropped} "
        f"reasoning_dropout_probability={reasoning_dropout_probability} "
        f"completion_loss_weight={completion_loss_weight} "
        f"completion_logits_mode={completion_logits_mode!r} "
        f"completion_projection_chunk_size={completion_projection_chunk_size} "
        f"sequential_objective_backward={sequential_objective_backward} "
        f"direct_loss_weight={direct_loss_weight} "
        f"pairwise_loss_weight={pairwise_loss_weight} "
        f"soft_loss_weight={soft_loss_weight} "
        f"mil_loss_weight={mil_loss_weight} "
        f"mil_pooling={mil_pooling!r} "
        f"mil_temperature={mil_temperature} "
        f"mil_top_k={mil_top_k} "
        f"mil_max_instances={mil_max_instances} "
        f"soft_loss_type={soft_loss_type!r} "
        f"direct_logits_mode={direct_logits_mode!r} "
        f"decision_head_mode={decision_head_mode!r} "
        f"decision_head_init={decision_head_init!r} "
        f"soft_target_logit_center={soft_target_logit_center} "
        f"soft_target_logit_scale={soft_target_logit_scale} "
        f"soft_huber_delta={soft_huber_delta} "
        f"ordinal_soft_loss_weight={ordinal_soft_loss_weight} "
        f"soft_teacher_artifact={soft_teacher_artifact} "
        f"pairwise_temperature={pairwise_temperature} "
        f"paired_batching={paired_batching} "
        f"paired_batching_mode={paired_batching_mode!r} "
        f"dataset_sampling={dataset_sampling!r} "
        f"train_sampling_strategy={train_sampling_strategy!r} "
        f"dataset_loss_weighting={dataset_loss_weighting!r} "
        f"group_dro_eta={group_dro_eta} "
        f"group_dro_ema={group_dro_ema} "
        f"expected_dataset_mass={expected_dataset_mass}"
    )
    rating_counts = Counter(
        record.get("rating") for record in records if record.get("rating") is not None
    )
    if rating_counts:
        print(f"selected_rating_counts={dict(sorted(rating_counts.items()))}")

    model_loader = str(
        OmegaConf.select(cfg, "student.model_loader", default="causal_lm")
    )
    model_loader_classes = {
        "causal_lm": AutoModelForCausalLM,
        "image_text_to_text": AutoModelForImageTextToText,
    }
    if model_loader not in model_loader_classes:
        raise ValueError(f"unknown student.model_loader={model_loader!r}")
    finetuning_mode = str(
        OmegaConf.select(cfg, "student.finetuning_mode", default="lora")
    )
    if finetuning_mode not in {"lora", "full"}:
        raise ValueError(f"unknown student.finetuning_mode={finetuning_mode!r}")
    quantization_enabled = bool(
        OmegaConf.select(cfg, "student.quantization.enabled", default=False)
    )
    gradient_checkpointing_requested = bool(
        OmegaConf.select(
            cfg,
            "student.training.gradient_checkpointing",
            default=True,
        )
    )
    gradient_checkpointing_policy = str(
        OmegaConf.select(
            cfg,
            "student.training.gradient_checkpointing_policy",
            default="all",
        )
    )
    raw_checkpointing_layer_indices = OmegaConf.select(
        cfg,
        "student.training.gradient_checkpointing_layer_indices",
        default=None,
    )
    checkpointing_layer_indices = (
        None
        if raw_checkpointing_layer_indices is None
        else [int(index) for index in raw_checkpointing_layer_indices]
    )
    if not gradient_checkpointing_requested and gradient_checkpointing_policy != "all":
        raise ValueError(
            "selective checkpointing policy requires gradient checkpointing"
        )
    lora_initialization = str(
        OmegaConf.select(cfg, "student.lora.init", default="default")
    )
    if lora_initialization not in {"default", "loftq", "eva"}:
        raise ValueError("student.lora.init must be default, loftq, or eva")
    lora_use_dora = bool(OmegaConf.select(cfg, "student.lora.use_dora", default=False))
    if lora_initialization == "loftq" and not quantization_enabled:
        raise ValueError("LoftQ initialization requires 4-bit quantization")
    eva_rho = float(OmegaConf.select(cfg, "student.lora.eva_rho", default=2.0))
    eva_tau = float(OmegaConf.select(cfg, "student.lora.eva_tau", default=0.99))
    eva_rows = int(OmegaConf.select(cfg, "student.lora.eva_rows", default=256))
    if eva_rows < 1:
        raise ValueError("student.lora.eva_rows must be positive")
    quantization_metadata: dict[str, Any] = {"enabled": quantization_enabled}
    model_kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16}
    if quantization_enabled:
        if model_loader != "causal_lm" or finetuning_mode != "lora":
            raise ValueError("4-bit QLoRA requires causal_lm LoRA training")
        if not torch.cuda.is_available():
            raise RuntimeError("4-bit QLoRA requires a CUDA device")
        quantization_type = str(
            OmegaConf.select(
                cfg,
                "student.quantization.bnb_4bit_quant_type",
                default="nf4",
            )
        )
        use_double_quant = bool(
            OmegaConf.select(
                cfg,
                "student.quantization.bnb_4bit_use_double_quant",
                default=True,
            )
        )
        compute_dtype = str(
            OmegaConf.select(
                cfg,
                "student.quantization.bnb_4bit_compute_dtype",
                default="bfloat16",
            )
        )
        if quantization_type != "nf4" or compute_dtype != "bfloat16":
            raise ValueError("standard QLoRA requires NF4 weights and bfloat16 compute")
        quantization_metadata.update(
            bnb_4bit_quant_type=quantization_type,
            bnb_4bit_use_double_quant=use_double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model_kwargs.update(
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=quantization_type,
                bnb_4bit_use_double_quant=use_double_quant,
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            device_map={"": torch.cuda.current_device()},
        )
    require_fla = bool(
        OmegaConf.select(
            cfg,
            "student.training.require_flash_linear_attention",
            default=False,
        )
    )
    fla_available = is_flash_linear_attention_available()
    if require_fla and not fla_available:
        raise RuntimeError(
            "Flash Linear Attention is required but unavailable; use the pinned "
            "FLA launcher environment"
        )
    print(
        f"flash_linear_attention_available={fla_available} required={require_fla}",
        flush=True,
    )
    require_causal_conv1d = bool(
        OmegaConf.select(
            cfg,
            "student.training.require_causal_conv1d",
            default=False,
        )
    ) or bool(os.environ.get("GLEIPNIR_CAUSAL_CONV1D_VERSION"))
    causal_conv1d_available = is_causal_conv1d_available()
    if require_causal_conv1d and not causal_conv1d_available:
        raise RuntimeError(
            "causal-conv1d is required but unavailable; use the pinned Qwen3.5 "
            "fast-kernel launcher environment"
        )
    print(
        f"causal_conv1d_available={causal_conv1d_available} "
        f"required={require_causal_conv1d}",
        flush=True,
    )
    require_flash_sdpa = bool(
        OmegaConf.select(
            cfg,
            "student.training.require_flash_sdpa",
            default=False,
        )
    )
    if require_flash_sdpa:
        if not torch.cuda.is_available():
            raise RuntimeError("required flash SDPA needs CUDA")
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
    print(
        "sdpa_backends="
        f"flash:{torch.backends.cuda.flash_sdp_enabled()},"
        f"math:{torch.backends.cuda.math_sdp_enabled()},"
        f"memory_efficient:{torch.backends.cuda.mem_efficient_sdp_enabled()},"
        f"cudnn:{torch.backends.cuda.cudnn_sdp_enabled()} "
        f"flash_required={require_flash_sdpa}",
        flush=True,
    )
    fla_prewarm_length_value = OmegaConf.select(
        cfg,
        "student.training.fla_prewarm_sequence_length",
        default=None,
    )
    fla_prewarm = None
    if fla_prewarm_length_value is not None:
        if not require_fla:
            raise ValueError("FLA prewarm requires flash-linear-attention")
        fla_prewarm = prewarm_gated_delta_rule(int(fla_prewarm_length_value))
        torch.cuda.empty_cache()
    model = model_loader_classes[model_loader].from_pretrained(
        str(cfg.student.model),
        **revision_kwargs,
        **model_kwargs,
    )
    if quantization_enabled:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=gradient_checkpointing_requested,
        )
    if decision_head_mode == "binary_head":
        if finetuning_mode != "lora" or model_loader != "causal_lm":
            raise ValueError("binary decision heads require causal LoRA training")
        hidden_size = int(model.config.hidden_size)
        decision_head = torch.nn.Linear(
            hidden_size,
            2,
            bias=decision_head_init == "random",
            device=model.device,
            # Match the causal LM head so its gradient enters the pinned FLA
            # backward kernel in the same dtype as the proven token-logit path.
            dtype=model.lm_head.weight.dtype,
        )
        if decision_head_init == "token_rows":
            if direct_target_ids is None:
                raise RuntimeError("binary token ids are unavailable")
            with torch.no_grad():
                decision_head.weight.copy_(
                    model.lm_head.weight.index_select(
                        0,
                        torch.tensor(
                            direct_target_ids,
                            device=model.lm_head.weight.device,
                            dtype=torch.long,
                        ),
                    ).to(decision_head.weight.dtype)
                )
        model.add_module("decision_head", decision_head)
    init_adapter_value = OmegaConf.select(cfg, "student.init_adapter", default=None)
    if finetuning_mode == "full" and init_adapter_value is not None:
        raise ValueError("student.init_adapter is incompatible with full fine-tuning")
    if finetuning_mode == "lora" and init_adapter_value is None:
        exclude_modules = OmegaConf.select(
            cfg, "student.lora.exclude_modules", default=None
        )
        lora_config_kwargs: dict[str, Any] = {
            "r": int(cfg.student.lora.r),
            "lora_alpha": int(cfg.student.lora.alpha),
            "lora_dropout": float(cfg.student.lora.dropout),
            "target_modules": list(cfg.student.lora.target_modules),
            "exclude_modules": (
                None if exclude_modules is None else str(exclude_modules)
            ),
            "task_type": "CAUSAL_LM",
            "use_dora": lora_use_dora,
            "modules_to_save": (
                ["decision_head"] if decision_head_mode == "binary_head" else None
            ),
        }
        if lora_initialization == "eva":
            lora_config_kwargs.update(
                init_lora_weights="eva",
                eva_config=EvaConfig(rho=eva_rho, tau=eva_tau),
            )
        model = get_peft_model(
            model,
            LoraConfig(**lora_config_kwargs),
        )
        if lora_initialization == "loftq":
            from huggingface_hub import try_to_load_from_cache

            cached_index = try_to_load_from_cache(
                str(cfg.student.model),
                "model.safetensors.index.json",
                revision=model_revision,
            )
            if not isinstance(cached_index, str):
                raise FileNotFoundError(
                    "LoftQ requires the cached model.safetensors.index.json"
                )
            initialized_loftq_modules = initialize_qwen35_loftq(
                model, Path(cached_index).parent
            )
            print(f"loftq_initialized_modules={initialized_loftq_modules}", flush=True)
        elif lora_initialization == "eva":
            generator = torch.Generator()
            generator.manual_seed(int(cfg.seed))
            indices = torch.randperm(len(tokenized), generator=generator)[
                : min(eva_rows, len(tokenized))
            ].tolist()
            eva_features = [
                {
                    "input_ids": tokenized[index]["direct_input_ids"],
                    "attention_mask": [1] * len(tokenized[index]["direct_input_ids"]),
                }
                for index in indices
            ]

            def collate_eva(features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
                return collate_eva_features(tokenizer, features)

            eva_loader = DataLoader(
                eva_features,
                batch_size=int(cfg.student.training.per_device_train_batch_size),
                shuffle=False,
                collate_fn=collate_eva,
            )
            initialize_lora_eva_weights(
                model,
                dataloader=eva_loader,
                show_progress_bar=True,
            )
    elif finetuning_mode == "lora":
        init_adapter = Path(str(init_adapter_value))
        if not init_adapter.is_absolute():
            init_adapter = root / init_adapter
        if not (init_adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(f"missing initial adapter: {init_adapter}")
        print(f"loading trainable initial adapter from {init_adapter}")
        model = PeftModel.from_pretrained(
            model,
            init_adapter.as_posix(),
            is_trainable=True,
        )
    kernel_modules = gated_delta_kernel_modules(model)
    if require_fla and (
        not kernel_modules
        or any(not module.startswith("fla.ops.") for module in kernel_modules)
    ):
        raise RuntimeError(
            f"Qwen gated-delta kernel did not resolve to FLA: {kernel_modules}"
        )
    causal_conv1d_modules = causal_conv1d_kernel_modules(model)
    if require_causal_conv1d and (
        not causal_conv1d_modules
        or any(
            not module.startswith("causal_conv1d.") for module in causal_conv1d_modules
        )
    ):
        raise RuntimeError(
            "Qwen causal convolution did not resolve to causal-conv1d: "
            f"{causal_conv1d_modules}"
        )
    trainable_lora_names = (
        validate_trainable_lora_layout(model, model_loader)
        if finetuning_mode == "lora"
        else []
    )
    counts = parameter_counts(model, finetuning_mode)
    trainable_dtypes = sorted(
        {
            str(parameter.dtype)
            for parameter in model.parameters()
            if parameter.requires_grad
        }
    )
    resolved_model = (
        model.get_base_model() if hasattr(model, "get_base_model") else model
    )
    print(
        f"model_loader={model_loader} "
        f"finetuning_mode={finetuning_mode} "
        f"resolved_model_class={resolved_model.__class__.__name__} "
        f"total_parameters={counts['total_parameters']} "
        f"trainable_parameters={counts['trainable_parameters']} "
        f"trainable_lora_tensors={len(trainable_lora_names)} "
        f"trainable_dtypes={trainable_dtypes} "
        f"gated_delta_kernel_modules={kernel_modules} "
        f"causal_conv1d_kernel_modules={causal_conv1d_modules} "
        f"first_trainable_lora="
        f"{trainable_lora_names[0] if trainable_lora_names else None}",
        flush=True,
    )

    output_dir = Path(str(cfg.student.output_dir))
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    torch_compile = bool(
        OmegaConf.select(cfg, "student.training.torch_compile", default=False)
    )
    selective_torch_compile_policy = str(
        OmegaConf.select(
            cfg,
            "student.training.selective_torch_compile_policy",
            default="none",
        )
    )
    selective_torch_compile_backend = str(
        OmegaConf.select(
            cfg,
            "student.training.selective_torch_compile_backend",
            default="inductor",
        )
    )
    selective_torch_compile_mode = str(
        OmegaConf.select(
            cfg,
            "student.training.selective_torch_compile_mode",
            default="default",
        )
    )
    selective_torch_compile_dynamic = bool(
        OmegaConf.select(
            cfg,
            "student.training.selective_torch_compile_dynamic",
            default=True,
        )
    )
    if torch_compile and selective_torch_compile_policy != "none":
        raise ValueError(
            "global and selective torch compilation cannot both be enabled"
        )
    selective_torch_compile_canary_tokens = int(
        OmegaConf.select(
            cfg,
            "student.training.selective_torch_compile_canary_tokens",
            default=0,
        )
    )
    selective_torch_compile_canary_atol = float(
        OmegaConf.select(
            cfg,
            "student.training.selective_torch_compile_canary_atol",
            default=0.05,
        )
    )
    selective_torch_compile_canary_rtol = float(
        OmegaConf.select(
            cfg,
            "student.training.selective_torch_compile_canary_rtol",
            default=0.01,
        )
    )
    if selective_torch_compile_canary_tokens < 0:
        raise ValueError("selective compile canary tokens cannot be negative")
    if (
        selective_torch_compile_canary_tokens
        and selective_torch_compile_policy == "none"
    ):
        raise ValueError("selective compile canary requires selective compilation")
    fsdp_enabled = bool(
        OmegaConf.select(cfg, "student.training.fsdp.enabled", default=False)
    )
    if fsdp_enabled and finetuning_mode != "full":
        raise ValueError("FSDP is reserved for full fine-tuning in this trainer")
    if fsdp_enabled:
        fsdp_config = OmegaConf.to_container(
            cfg.student.training.fsdp.config, resolve=True
        )
    else:
        fsdp_config = None
    gradient_checkpointing_enabled = (
        gradient_checkpointing_requested and not fsdp_enabled
    )
    trainer_manages_gradient_checkpointing = (
        trainer_should_manage_gradient_checkpointing(
            gradient_checkpointing_enabled,
            gradient_checkpointing_policy,
        )
    )
    if gradient_checkpointing_enabled:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        checkpointed_layer_indices = apply_gradient_checkpointing_policy(
            model,
            gradient_checkpointing_policy,
            checkpointing_layer_indices,
        )
    else:
        checkpointed_layer_indices = []
    save_strategy = str(
        OmegaConf.select(cfg, "student.training.save_strategy", default="no")
    )
    if save_strategy not in {"no", "steps", "epoch"}:
        raise ValueError("student.training.save_strategy must be no, steps, or epoch")
    save_steps = float(
        OmegaConf.select(cfg, "student.training.save_steps", default=500)
    )
    if save_steps <= 0:
        raise ValueError("student.training.save_steps must be positive")
    if save_steps >= 1 and not save_steps.is_integer():
        raise ValueError("student.training.save_steps >= 1 must be an integer")
    save_total_limit = int(
        OmegaConf.select(cfg, "student.training.save_total_limit", default=1)
    )
    if save_total_limit < 1:
        raise ValueError("student.training.save_total_limit must be positive")
    save_only_model = bool(
        OmegaConf.select(cfg, "student.training.save_only_model", default=True)
    )
    trainer_optim = str(cfg.student.training.optim)
    args = TrainingArguments(
        output_dir=output_dir.as_posix(),
        optim=trainer_optim,
        num_train_epochs=float(cfg.student.training.num_train_epochs),
        max_steps=int(cfg.student.training.max_steps),
        learning_rate=float(cfg.student.training.learning_rate),
        lr_scheduler_type=str(
            OmegaConf.select(
                cfg, "student.training.lr_scheduler_type", default="linear"
            )
        ),
        warmup_steps=training_warmup_steps(cfg.student.training),
        weight_decay=float(cfg.student.training.weight_decay),
        per_device_train_batch_size=int(
            cfg.student.training.per_device_train_batch_size
        ),
        gradient_accumulation_steps=int(
            cfg.student.training.gradient_accumulation_steps
        ),
        torch_compile=torch_compile,
        # TrainingArguments silently enables compilation when either option is
        # non-null, even if torch_compile=False. Keep eager runs genuinely eager.
        torch_compile_backend=(
            str(
                OmegaConf.select(
                    cfg, "student.training.torch_compile_backend", default="inductor"
                )
            )
            if torch_compile
            else None
        ),
        torch_compile_mode=(
            str(
                OmegaConf.select(
                    cfg, "student.training.torch_compile_mode", default="default"
                )
            )
            if torch_compile
            else None
        ),
        logging_steps=int(cfg.student.training.logging_steps),
        seed=int(cfg.seed),
        data_seed=int(cfg.seed),
        train_sampling_strategy=train_sampling_strategy,
        length_column_name="length",
        save_strategy=save_strategy,
        save_steps=int(save_steps) if save_steps >= 1 else save_steps,
        save_total_limit=save_total_limit,
        save_only_model=save_only_model,
        bf16=True,
        # Trainer calls gradient_checkpointing_enable() again at train startup.
        # For selective policies that would silently reset every layer to True.
        gradient_checkpointing=trainer_manages_gradient_checkpointing,
        fsdp=True if fsdp_enabled else None,
        fsdp_config=fsdp_config,
        report_to="none",
        remove_unused_columns=False,
    )
    optimizer_name = str(cfg.student.training.optimizer)
    if optimizer_name not in {"adamw", "muon"}:
        raise ValueError(f"unknown student.training.optimizer={optimizer_name!r}")
    trainer_cls = (
        MuonAuxiliarySFTTrainer if optimizer_name == "muon" else AuxiliarySFTTrainer
    )
    collator = CompletionOnlyCollator(tokenizer.pad_token_id)
    optimizer_step_timer = OptimizerStepTimer()
    trainer = trainer_cls(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[optimizer_step_timer],
    )
    if gradient_checkpointing_enabled:
        checkpointed_layer_indices = apply_gradient_checkpointing_policy(
            model,
            gradient_checkpointing_policy,
            checkpointing_layer_indices,
        )
    selective_torch_compile_canary = None
    if selective_torch_compile_canary_tokens:
        if direct_target_ids is None:
            raise ValueError("selective compile canary requires direct target ids")
        canary_ids = list(dataset[0]["direct_input_ids"])[
            -selective_torch_compile_canary_tokens:
        ]
        canary_device = next(
            parameter.device
            for parameter in model.parameters()
            if parameter.device.type != "meta"
        )
        canary_input_ids = torch.tensor(
            [canary_ids], dtype=torch.long, device=canary_device
        )
        canary_attention_mask = torch.ones_like(canary_input_ids)
        model.eval()
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=canary_device.type,
                dtype=torch.bfloat16,
                enabled=canary_device.type == "cuda",
            ),
        ):
            eager_canary_logits, _ = forward_final_token_logits(
                model,
                canary_input_ids,
                canary_attention_mask,
                direct_logits_mode,
            )
            decision_ids = torch.tensor(
                direct_target_ids,
                dtype=torch.long,
                device=eager_canary_logits.device,
            )
            eager_canary_logits = eager_canary_logits.index_select(-1, decision_ids)
        selectively_compiled_layer_indices = apply_selective_torch_compile_policy(
            model,
            selective_torch_compile_policy,
            backend=selective_torch_compile_backend,
            mode=selective_torch_compile_mode,
            dynamic=selective_torch_compile_dynamic,
        )
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=canary_device.type,
                dtype=torch.bfloat16,
                enabled=canary_device.type == "cuda",
            ),
        ):
            compiled_canary_logits, _ = forward_final_token_logits(
                model,
                canary_input_ids,
                canary_attention_mask,
                direct_logits_mode,
            )
            compiled_canary_logits = compiled_canary_logits.index_select(
                -1, decision_ids
            )
        selective_torch_compile_canary = {
            "tokens": len(canary_ids),
            "autocast_dtype": "bfloat16",
            **compare_compile_canary_logits(
                eager_canary_logits,
                compiled_canary_logits,
                absolute_tolerance=selective_torch_compile_canary_atol,
                relative_tolerance=selective_torch_compile_canary_rtol,
            ),
        }
        if not selective_torch_compile_canary["passed"]:
            raise ValueError(
                "selective compilation logit canary failed: "
                f"{selective_torch_compile_canary}"
            )
        model.train()
    else:
        selectively_compiled_layer_indices = apply_selective_torch_compile_policy(
            model,
            selective_torch_compile_policy,
            backend=selective_torch_compile_backend,
            mode=selective_torch_compile_mode,
            dynamic=selective_torch_compile_dynamic,
        )
    compile_base = model.get_base_model() if hasattr(model, "get_base_model") else model
    compile_layer_types = getattr(compile_base.config, "layer_types", [])
    disabled_linear_attention_layer_indices = [
        index
        for index, layer_type in enumerate(compile_layer_types)
        if selective_torch_compile_policy
        in {
            "full_attention_and_linear_shell",
            "checkpointed_full_attention_and_linear_shell",
            "decoder_shells_without_token_mixers",
            "linear_attention_shells_only",
        }
        and layer_type == "linear_attention"
    ]
    disabled_linear_attention_kernel_layer_indices = [
        index
        for index, layer_type in enumerate(compile_layer_types)
        if selective_torch_compile_policy == "full_attention_and_linear_mixer_segments"
        and layer_type == "linear_attention"
    ]
    disabled_linear_attention_kernel_components = (
        ["causal_conv1d_fn", "chunk_gated_delta_rule", "norm.forward"]
        if selective_torch_compile_policy == "full_attention_and_linear_mixer_segments"
        else []
    )
    disabled_full_attention_layer_indices = [
        index
        for index, layer_type in enumerate(compile_layer_types)
        if selective_torch_compile_policy == "decoder_shells_without_token_mixers"
        and layer_type == "full_attention"
    ]
    trainer.direct_target_ids = direct_target_ids
    train_output = trainer.train()
    train_metrics = {
        key: value.item() if hasattr(value, "item") else value
        for key, value in train_output.metrics.items()
    }
    checkpoints = sorted(
        (
            path
            for path in output_dir.glob("checkpoint-*")
            if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()
        ),
        key=lambda path: int(path.name.removeprefix("checkpoint-")),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir.as_posix())
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output_dir)
        (output_dir / "training_metadata.json").write_text(
            json.dumps(
                {
                    **counts,
                    "finetuning_mode": finetuning_mode,
                    "model_loader": model_loader,
                    "model": str(cfg.student.model),
                    "model_revision": model_revision,
                    "commit": os.environ.get("GLEIPNIR_COMMIT"),
                    "seed": int(cfg.seed),
                    "optimization": {
                        "optimizer": optimizer_name,
                        "trainer_optim": trainer_optim,
                        "learning_rate": float(cfg.student.training.learning_rate),
                        "lr_scheduler_type": str(
                            OmegaConf.select(
                                cfg,
                                "student.training.lr_scheduler_type",
                                default="linear",
                            )
                        ),
                        "warmup_steps": training_warmup_steps(cfg.student.training),
                        "weight_decay": float(cfg.student.training.weight_decay),
                        "dataset_sampling": dataset_sampling,
                        "dataset_loss_weighting": dataset_loss_weighting,
                        "group_dro_eta": group_dro_eta,
                        "group_dro_ema": group_dro_ema,
                        "group_dro_state": (
                            None
                            if group_dro_loss is None
                            else group_dro_loss.snapshot()
                        ),
                        "expected_dataset_mass": expected_dataset_mass,
                        "lora_dropout": (
                            float(cfg.student.lora.dropout)
                            if finetuning_mode == "lora"
                            else None
                        ),
                        "lora_initialization": (
                            lora_initialization if finetuning_mode == "lora" else None
                        ),
                        "lora_use_dora": (
                            lora_use_dora if finetuning_mode == "lora" else None
                        ),
                        "lora_target_modules": (
                            list(cfg.student.lora.target_modules)
                            if finetuning_mode == "lora"
                            else None
                        ),
                        "eva_rho": (
                            eva_rho
                            if finetuning_mode == "lora"
                            and lora_initialization == "eva"
                            else None
                        ),
                        "eva_tau": (
                            eva_tau
                            if finetuning_mode == "lora"
                            and lora_initialization == "eva"
                            else None
                        ),
                        "eva_rows": (
                            min(eva_rows, len(tokenized))
                            if finetuning_mode == "lora"
                            and lora_initialization == "eva"
                            else None
                        ),
                        "soft_target_logit_scale": soft_target_logit_scale,
                        "save_strategy": save_strategy,
                        "save_steps": (
                            int(save_steps) if save_steps >= 1 else save_steps
                        ),
                        "save_total_limit": save_total_limit,
                        "save_only_model": save_only_model,
                        "muon_adjust_lr_fn": (
                            str(cfg.student.training.muon_adjust_lr_fn)
                            if optimizer_name == "muon"
                            else None
                        ),
                        "muon_momentum": (
                            float(cfg.student.training.muon_momentum)
                            if optimizer_name == "muon"
                            else None
                        ),
                        "muon_nesterov": (
                            bool(cfg.student.training.muon_nesterov)
                            if optimizer_name == "muon"
                            else None
                        ),
                        "muon_ns_steps": (
                            int(cfg.student.training.muon_ns_steps)
                            if optimizer_name == "muon"
                            else None
                        ),
                    },
                    "direct_logits_mode": direct_logits_mode,
                    "decision_head_mode": decision_head_mode,
                    "decision_head_init": decision_head_init,
                    "decision_head_gradient_mode": (
                        "lm_token_straight_through"
                        if decision_head_mode == "binary_head"
                        else None
                    ),
                    "losses": {
                        "completion_weight": completion_loss_weight,
                        "completion_logits_mode": completion_logits_mode,
                        "completion_projection_chunk_size": (
                            completion_projection_chunk_size
                        ),
                        "sequential_objective_backward": (
                            sequential_objective_backward
                        ),
                        "direct_weight": direct_loss_weight,
                        "pairwise_weight": pairwise_loss_weight,
                        "soft_weight": soft_loss_weight,
                        "soft_type": soft_loss_type,
                        "mil_weight": mil_loss_weight,
                        "mil_pooling": mil_pooling,
                        "mil_temperature": mil_temperature,
                        "mil_top_k": mil_top_k,
                        "mil_max_instances": mil_max_instances,
                    },
                    "quantization": quantization_metadata,
                    "flash_linear_attention": {
                        "available": fla_available,
                        "required": require_fla,
                        "version": os.environ.get("GLEIPNIR_FLA_VERSION"),
                        "backend_dispatch_disabled": os.environ.get(
                            "FLA_DISABLE_BACKEND_DISPATCH"
                        )
                        == "1",
                        "triton_version": os.environ.get("GLEIPNIR_TRITON_VERSION"),
                        "in_process_prewarm": fla_prewarm,
                    },
                    "gated_delta_kernel_modules": kernel_modules,
                    "causal_conv1d": {
                        "available": causal_conv1d_available,
                        "required": require_causal_conv1d,
                        "version": os.environ.get("GLEIPNIR_CAUSAL_CONV1D_VERSION"),
                        "kernel_modules": causal_conv1d_modules,
                    },
                    "sdpa": {
                        "flash_required": require_flash_sdpa,
                        "flash_enabled": torch.backends.cuda.flash_sdp_enabled(),
                        "math_enabled": torch.backends.cuda.math_sdp_enabled(),
                        "memory_efficient_enabled": (
                            torch.backends.cuda.mem_efficient_sdp_enabled()
                        ),
                        "cudnn_enabled": torch.backends.cuda.cudnn_sdp_enabled(),
                    },
                    "training_batch": {
                        "micro_batch_size": int(
                            cfg.student.training.per_device_train_batch_size
                        ),
                        "gradient_accumulation_steps": int(
                            cfg.student.training.gradient_accumulation_steps
                        ),
                        "effective_batch_size": int(
                            cfg.student.training.per_device_train_batch_size
                        )
                        * int(cfg.student.training.gradient_accumulation_steps),
                    },
                    "gradient_checkpointing": gradient_checkpointing_enabled,
                    "trainer_manages_gradient_checkpointing": (
                        trainer_manages_gradient_checkpointing
                    ),
                    "gradient_checkpointing_policy": gradient_checkpointing_policy,
                    "gradient_checkpointing_layer_indices": (
                        checkpointing_layer_indices
                    ),
                    "checkpointed_layer_indices": checkpointed_layer_indices,
                    "selective_torch_compile": {
                        "policy": selective_torch_compile_policy,
                        "compiled_layer_indices": selectively_compiled_layer_indices,
                        "disabled_linear_attention_layer_indices": (
                            disabled_linear_attention_layer_indices
                        ),
                        "disabled_linear_attention_kernel_layer_indices": (
                            disabled_linear_attention_kernel_layer_indices
                        ),
                        "disabled_linear_attention_kernel_components": (
                            disabled_linear_attention_kernel_components
                        ),
                        "disabled_full_attention_layer_indices": (
                            disabled_full_attention_layer_indices
                        ),
                        "disabled_full_attention_components": (
                            ["self_attn.forward"]
                            if disabled_full_attention_layer_indices
                            else []
                        ),
                        "backend": selective_torch_compile_backend,
                        "mode": selective_torch_compile_mode,
                        "dynamic": selective_torch_compile_dynamic,
                        "global_torch_compile": torch_compile,
                        "dynamo_counters": torch_compile_counter_snapshot(),
                        "canary": selective_torch_compile_canary,
                    },
                    "materialized_training_inputs": {
                        "completion": bool(completion_loss_weight),
                        "direct": uses_direct_forward,
                    },
                    "sequence_lengths": {
                        "completion_max_length": completion_max_length,
                        "direct_max_length": max_length,
                    },
                    "batching": {
                        "train_sampling_strategy": train_sampling_strategy,
                        "length_column_name": "length",
                        "padding": collator.padding_statistics(),
                    },
                    "optimizer_step_timing": optimizer_step_timer.summary(),
                    "training_state": {
                        "global_step": int(trainer.state.global_step),
                        "max_steps": int(trainer.state.max_steps),
                        "retained_checkpoints": [path.name for path in checkpoints],
                        "log_history": trainer.state.log_history,
                    },
                    "train_metrics": train_metrics,
                    "peak_cuda_memory_allocated_bytes": (
                        torch.cuda.max_memory_allocated()
                        if torch.cuda.is_available()
                        else None
                    ),
                    "peak_cuda_memory_reserved_bytes": (
                        torch.cuda.max_memory_reserved()
                        if torch.cuda.is_available()
                        else None
                    ),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        (output_dir.parent / "config.yaml").write_text(
            OmegaConf.to_yaml(cfg, resolve=True)
        )
        print(f"saved {finetuning_mode} model to {output_dir}")


if __name__ == "__main__":
    main()
