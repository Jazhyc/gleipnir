"""Freeze strict teacher agreement before a new monitoring campaign."""

from __future__ import annotations

import copy
import fcntl
import json
import math
from collections import Counter

import yaml

from gleipnir.data.injection_audit import audit_rows
from gleipnir.data.monitoring import file_hash, read_rows, validate_targets, write_json
from gleipnir.data.teacher_filter import remove_disagreement

from .contract import Campaign


def key(row: dict) -> tuple[str, str]:
    return row["dataset"], str(row["index"])


def retained_indices(rows: list[dict], targets: list[dict]) -> list[int]:
    """Join by identity and prompt provenance, never teacher-cache order."""
    validate_targets(rows, targets)
    teachers = {key(row): row for row in targets}
    return [
        i
        for i, row in enumerate(rows)
        if not remove_disagreement(row["label"], teachers[key(row)]["soft_target"], 0)
    ]


def token_lengths(ctx: Campaign, rows: list[dict]) -> list[int]:
    """Recount actual untruncated decision-boundary inputs on CPU."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        ctx.root / ctx.config["base_model"], local_files_only=True
    )
    lengths = []
    for start in range(0, len(rows), 64):
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": row["student_prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            + "Prediction:"
            for row in rows[start : start + 64]
        ]
        lengths.extend(
            map(len, tokenizer(prompts, add_special_tokens=False)["input_ids"])
        )
    return lengths


def resolve_filter(ctx: Campaign) -> Campaign:
    """Resolve new campaigns; preserve original configs and prepared history."""
    policy = ctx.config.get("teacher_filter", "strict")
    if policy not in ("strict", "none"):
        raise ValueError("teacher_filter must be strict or none")
    if policy == "none":
        return ctx
    destination = ctx.data / "strict_filter"
    resolved = destination / "config.yaml"
    if ctx.config_path == resolved:
        return ctx
    if not resolved.exists() and (ctx.data / "manifest.json").exists():
        return ctx
    with_lock = ctx.output / "filter.lock"
    with_lock.parent.mkdir(parents=True, exist_ok=True)
    with with_lock.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if resolved.exists():
            candidate = Campaign.load(ctx.root, resolved)
            if candidate.config["inputs"]["unfiltered_config"]["sha256"] != file_hash(
                ctx.config_path
            ):
                raise ValueError("unfiltered campaign configuration changed")
            return candidate
        inventory = ctx.inventory()
        if not inventory["passed"]:
            raise ValueError(
                "campaign prerequisites failed: " + "; ".join(inventory["errors"])
            )
        rows, targets = (
            read_rows(ctx.input("training")),
            read_rows(ctx.input("teacher")),
        )
        if len(rows) != ctx.config["training_rows"]:
            raise ValueError("unfiltered training population changed")
        retained = retained_indices(rows, targets)
        if not retained:
            raise ValueError("strict filtering removes every training row")
        selected = [rows[i] for i in retained]
        augmentation = None
        if ctx.config.get("augmentation_audit", False):
            augmentation = audit_rows(
                read_rows(ctx.input("clean_training")),
                rows,
                read_rows(ctx.input("ledger")),
                json.loads(ctx.input("templates").read_text()),
            )
            if augmentation["injected_rows"] != ctx.config["injected_rows"]:
                raise ValueError("augmentation coverage changed before filtering")
        lengths = token_lengths(ctx, selected)
        cap = ctx.job()["max_length"]
        if max(lengths) > cap:
            raise ValueError("filtered input exceeds context cap; no truncation")
        keep = {key(row) for row in selected}
        retained_set = set(retained)
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("training", "teacher"):
            # Preserve retained bytes/order, including the teacher's cache order.
            with (
                ctx.input(name).open() as source,
                (destination / f"{name}.jsonl").open("w") as output,
            ):
                for line in source:
                    if line.strip() and key(json.loads(line)) in keep:
                        output.write(line)
        audit = {
            "rows": len(selected),
            "total": sum(lengths),
            "maximum": max(lengths),
            "configured_cap": cap,
            "truncated": 0,
        }
        write_json(destination / "token_audit.json", audit)
        receipt = {
            "policy": "strict",
            "threshold": 0.5,
            "delta": 0,
            "removed": len(rows) - len(selected),
            "retained": len(selected),
            "removed_ids": [
                key(row) for i, row in enumerate(rows) if i not in retained_set
            ],
            "retained_by_source_label": dict(
                Counter(f"{r['dataset']}/{r['label']}" for r in selected)
            ),
            "removed_by_source_label": dict(
                Counter(
                    f"{r['dataset']}/{r['label']}"
                    for i, r in enumerate(rows)
                    if i not in retained_set
                )
            ),
            "original_augmentation_audit": augmentation,
        }
        write_json(destination / "filter.json", receipt)
        config = copy.deepcopy(ctx.config)
        config.update(
            teacher_filter="strict",
            training_rows=len(selected),
            expected_steps=math.ceil(len(selected) / config["logical_batch_size"]),
            expected_training_tokens=sum(lengths),
            augmentation_audit=False,
        )
        config["intervention"] += (
            " Apply strict teacher/source-label agreement filtering at threshold 0.5 "
            "before training; retain original soft targets."
        )
        # The original augmentation population was validated above, before removal.
        for name in ("training", "teacher", "token_audit"):
            config["inputs"]["unfiltered_" + name] = config["inputs"][name]
        for name, filename in [
            ("training", "training.jsonl"),
            ("teacher", "teacher.jsonl"),
            ("token_audit", "token_audit.json"),
            ("filter", "filter.json"),
        ]:
            path = destination / filename
            config["inputs"][name] = {
                "path": str(path.relative_to(ctx.root)),
                "sha256": file_hash(path),
            }
        config["inputs"]["unfiltered_config"] = {
            "path": str(
                ctx.config_path.relative_to(ctx.root)
                if ctx.config_path.is_relative_to(ctx.root)
                else ctx.config_path
            ),
            "sha256": file_hash(ctx.config_path),
        }
        resolved.write_text(yaml.safe_dump(config, sort_keys=False))
        print(
            f"strict_teacher_filter removed={receipt['removed']} "
            f"retained={len(selected)}",
            flush=True,
        )
        return Campaign.load(ctx.root, resolved)
