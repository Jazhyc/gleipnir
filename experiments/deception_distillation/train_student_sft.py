#!/usr/bin/env python3
"""SFT a Qwen LoRA on hard, rationale, or soft monitor targets."""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import SequentialSampler

from gleipnir.training import (
    MuonAdamW,
    muon_adamw_param_groups,
)


class CompletionOnlyCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        width = max(len(feature["input_ids"]) for feature in features)
        input_ids, attention_mask, labels = [], [], []
        for feature in features:
            padding = width - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * padding)
            attention_mask.append([1] * len(feature["input_ids"]) + [0] * padding)
            labels.append(feature["labels"] + [-100] * padding)
        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        if "direct_input_ids" in features[0]:
            direct_width = max(len(feature["direct_input_ids"]) for feature in features)
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
        return batch


REASONING_BLOCK_START = "\n\n<assistant_reasoning>\n"
REASONING_BLOCK_END = "\n</assistant_reasoning>"
SOURCE_PROMPT_TEMPLATE_KEY = "_source_prompt_template"
LABEL_ONLY_TARGET_KEY = "_label_only_target"
DIRECT_PREDICTION_PREFIX = "Prediction:"
DIRECT_RATING_PREFIX = "Rating:"
SOFT_TARGET_KEY = "_soft_target"
CANONICAL_QWEN35_LORA_FRAGMENT = ".model.language_model.layers."
VISION_MODULE_MARKERS = ("visual", "vision_tower", "merger", "patch_embed")


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
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    lora_trainable = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" in name
    )
    if finetuning_mode == "lora":
        if not lora_trainable or trainable != lora_trainable:
            raise RuntimeError(
                "LoRA mode must train only non-empty LoRA parameters: "
                f"trainable={trainable} lora_trainable={lora_trainable}"
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
    }


def gated_delta_kernel_modules(model: torch.nn.Module) -> list[str]:
    """Return the concrete Qwen gated-delta implementations bound to the model."""
    modules = {
        kernel.__module__
        for module in model.modules()
        if (kernel := getattr(module, "chunk_gated_delta_rule", None)) is not None
    }
    return sorted(modules)


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
            attention_mask=attention_mask,
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
        attention_mask=attention_mask,
        use_cache=False,
        logits_to_keep=selected_positions,
    )
    if outputs.logits.shape[1] != len(selected_positions):
        raise RuntimeError(
            "model did not honor logits_to_keep: "
            f"requested={len(selected_positions)} got={outputs.logits.shape[1]}"
        )
    return outputs.logits[row_indices, inverse], outputs


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
        return F.binary_cross_entropy_with_logits(student_margins, targets)

    teacher_margins = (torch.logit(targets) - float(target_logit_center)) / float(
        target_logit_scale
    )
    if loss_type == "bce":
        transformed_targets = torch.sigmoid(teacher_margins)
        return F.binary_cross_entropy_with_logits(
            student_margins,
            transformed_targets,
        )
    return F.smooth_l1_loss(
        student_margins,
        teacher_margins,
        beta=float(huber_delta),
    )


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
    prompt_template: str | None = None,
    prompt_template_without_reasoning: str | None = None,
    target_mode: str = "teacher",
    reasoning_dropout_probability: float = 0.0,
    reasoning_dropout_seed: int = 0,
    include_direct_target: bool = False,
    direct_target_prefix: str = DIRECT_PREDICTION_PREFIX,
    dataset_id: int | None = None,
) -> dict[str, Any]:
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
    effective_target_mode = (
        "prediction_only" if record.get(LABEL_ONLY_TARGET_KEY) else target_mode
    )
    if effective_target_mode == "teacher":
        target = record["student_target"]
    elif effective_target_mode == "prediction_only":
        target = f"Prediction:{int(record['label'])}"
    else:
        raise ValueError(f"unknown student.target_mode={effective_target_mode!r}")
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": raw_prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    target_ids = tokenizer.encode(
        target + (tokenizer.eos_token or ""),
        add_special_tokens=False,
    )
    if len(target_ids) >= max_length:
        raise ValueError(
            f"target alone exceeds student.max_length for index={record['index']}"
        )
    prompt_ids = prompt_ids[-(max_length - len(target_ids)) :]
    tokenized: dict[str, Any] = {
        "input_ids": prompt_ids + target_ids,
        "labels": [-100] * len(prompt_ids) + target_ids,
    }
    if include_direct_target:
        if dataset_id is None:
            raise ValueError("dataset_id is required for direct-target training")
        direct_ids = tokenizer.encode(
            prompt + direct_target_prefix,
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
        LoraConfig,
        PeftModel,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )
    from transformers.utils.import_utils import is_flash_linear_attention_available

    direct_loss_weight = float(
        OmegaConf.select(cfg, "student.training.direct_loss_weight", default=0.0)
    )
    completion_loss_weight = float(
        OmegaConf.select(cfg, "student.training.completion_loss_weight", default=1.0)
    )
    pairwise_loss_weight = float(
        OmegaConf.select(cfg, "student.training.pairwise_loss_weight", default=0.0)
    )
    soft_loss_weight = float(
        OmegaConf.select(cfg, "student.training.soft_loss_weight", default=0.0)
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
            "student.training.direct_logits_mode must be full or "
            "selected_positions"
        )
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
    if any(
        weight < 0
        for weight in (
            completion_loss_weight,
            direct_loss_weight,
            pairwise_loss_weight,
            soft_loss_weight,
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
            ordinal_soft_loss_weight,
        )
    ):
        raise ValueError("at least one student loss weight must be positive")
    if ordinal_soft_loss_weight and (
        direct_loss_weight or pairwise_loss_weight or soft_loss_weight
    ):
        raise ValueError(
            "ordinal soft loss cannot be combined with binary auxiliary losses"
        )
    if pairwise_temperature <= 0:
        raise ValueError("student.training.pairwise_temperature must be positive")
    if soft_loss_type not in {"bce", "huber"}:
        raise ValueError("student.training.soft_loss_type must be one of: bce, huber")
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
    if paired_batching_mode not in {"interleaved", "same_dataset"}:
        raise ValueError(
            "student.training.paired_batching_mode must be interleaved or same_dataset"
        )
    uses_direct_forward = bool(
        direct_loss_weight
        or pairwise_loss_weight
        or soft_loss_weight
        or ordinal_soft_loss_weight
    )

    class AuxiliarySFTTrainer(Trainer):
        """Completion SFT with optional direct-label and within-dataset rank losses."""

        def _get_train_sampler(self, train_dataset=None):
            if paired_batching:
                selected_dataset = (
                    self.train_dataset if train_dataset is None else train_dataset
                )
                return SequentialSampler(selected_dataset)
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
            outputs = None
            loss = None
            if completion_loss_weight:
                outputs = model(**inputs)
                loss = completion_loss_weight * outputs.loss
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
                next_logits, direct_outputs = forward_final_token_logits(
                    model,
                    direct_input_ids,
                    direct_attention_mask,
                    direct_logits_mode,
                )
                label_ids = torch.tensor(
                    self.direct_target_ids,
                    device=next_logits.device,
                    dtype=torch.long,
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
                    loss = loss + soft_loss_weight * soft_binary_distillation_loss(
                        direct_logits,
                        soft_targets,
                        loss_type=soft_loss_type,
                        target_logit_center=soft_target_logit_center,
                        target_logit_scale=soft_target_logit_scale,
                        huber_delta=soft_huber_delta,
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
                    muon_adjust_lr_fn=str(
                        cfg.student.training.muon_adjust_lr_fn
                    ),
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
    tokenizer = AutoTokenizer.from_pretrained(str(cfg.student.model))
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
    tokenized = [
        tokenize_record(
            record,
            tokenizer,
            int(cfg.student.max_length),
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
            direct_target_prefix=direct_target_prefix,
            dataset_id=dataset_id_by_name[str(record.get("dataset", ""))],
        )
        for record in records
    ]
    dataset = Dataset.from_list(tokenized)
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
        f"direct_loss_weight={direct_loss_weight} "
        f"pairwise_loss_weight={pairwise_loss_weight} "
        f"soft_loss_weight={soft_loss_weight} "
        f"soft_loss_type={soft_loss_type!r} "
        f"direct_logits_mode={direct_logits_mode!r} "
        f"soft_target_logit_center={soft_target_logit_center} "
        f"soft_target_logit_scale={soft_target_logit_scale} "
        f"soft_huber_delta={soft_huber_delta} "
        f"ordinal_soft_loss_weight={ordinal_soft_loss_weight} "
        f"soft_teacher_artifact={soft_teacher_artifact} "
        f"pairwise_temperature={pairwise_temperature} "
        f"paired_batching={paired_batching} "
        f"paired_batching_mode={paired_batching_mode!r}"
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
            raise ValueError(
                "standard QLoRA requires NF4 weights and bfloat16 compute"
            )
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
        f"flash_linear_attention_available={fla_available} "
        f"required={require_fla}",
        flush=True,
    )
    model = model_loader_classes[model_loader].from_pretrained(
        str(cfg.student.model),
        **model_kwargs,
    )
    if quantization_enabled:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )
    init_adapter_value = OmegaConf.select(cfg, "student.init_adapter", default=None)
    if finetuning_mode == "full" and init_adapter_value is not None:
        raise ValueError("student.init_adapter is incompatible with full fine-tuning")
    if finetuning_mode == "lora" and init_adapter_value is None:
        exclude_modules = OmegaConf.select(
            cfg, "student.lora.exclude_modules", default=None
        )
        model = get_peft_model(
            model,
            LoraConfig(
                r=int(cfg.student.lora.r),
                lora_alpha=int(cfg.student.lora.alpha),
                lora_dropout=float(cfg.student.lora.dropout),
                target_modules=list(cfg.student.lora.target_modules),
                exclude_modules=(
                    None if exclude_modules is None else str(exclude_modules)
                ),
                task_type="CAUSAL_LM",
            ),
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
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    args = TrainingArguments(
        output_dir=output_dir.as_posix(),
        optim=str(cfg.student.training.optim),
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
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=not fsdp_enabled,
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
    trainer = trainer_cls(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=CompletionOnlyCollator(tokenizer.pad_token_id),
    )
    trainer.direct_target_ids = direct_target_ids
    train_output = trainer.train()
    train_metrics = {
        key: value.item() if hasattr(value, "item") else value
        for key, value in train_output.metrics.items()
    }
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
                    "commit": os.environ.get("GLEIPNIR_COMMIT"),
                    "seed": int(cfg.seed),
                    "optimization": {
                        "optimizer": optimizer_name,
                        "learning_rate": float(cfg.student.training.learning_rate),
                        "lr_scheduler_type": str(
                            OmegaConf.select(
                                cfg,
                                "student.training.lr_scheduler_type",
                                default="linear",
                            )
                        ),
                        "warmup_steps": training_warmup_steps(
                            cfg.student.training
                        ),
                        "weight_decay": float(cfg.student.training.weight_decay),
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
                    "quantization": quantization_metadata,
                    "flash_linear_attention": {
                        "available": fla_available,
                        "required": require_fla,
                        "version": os.environ.get("GLEIPNIR_FLA_VERSION"),
                    },
                    "gated_delta_kernel_modules": kernel_modules,
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
