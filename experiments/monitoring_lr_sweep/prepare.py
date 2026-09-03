#!/usr/bin/env python3
"""Validate frozen inputs and materialize the monitoring-only LR job manifest."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.monitoring_lr_sweep.core import (
    EFFECTIVE_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATES,
    LORA_ALPHA,
    MAX_LENGTH,
    MICRO_BATCH_SIZE,
    MODEL_ID,
    MODEL_REVISION,
    RANK,
    SOFT_TARGETS_SHA256,
    STUDENT_ROWS_SHA256,
    TRAIN_ROWS,
    TRAINING_SEED,
    job_name,
    validate_jobs,
)

DEFAULT_DATA_DIR = Path("data/tool_trajectory_monitoring/distillation_scaling")
DEFAULT_ID_INPUT = Path("data/tool_trajectory_monitoring/distillation_id/prompts.jsonl")
DEFAULT_RESULT_DIR = Path("results/monitoring_lr_sweep")
ID_INPUT_SHA256 = "ecc8b204be2505d2600c38d1747b87e2ef3c8a53ecd51908a79ad8e86e102389"
EXPECTED_SOURCES = {"bash_arena", "bash_bench", "cot_red_handed", "gloom", "stride"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def inspect_training_inputs(
    student_path: Path, soft_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if sha256_file(student_path) != STUDENT_ROWS_SHA256:
        raise ValueError("student-row checksum drifted")
    if sha256_file(soft_path) != SOFT_TARGETS_SHA256:
        raise ValueError("soft-target checksum drifted")
    students = read_jsonl(student_path)
    targets = read_jsonl(soft_path)
    if len(students) != TRAIN_ROWS or len(targets) != TRAIN_ROWS:
        raise ValueError("monitoring-only training row count drifted")
    strata: Counter[tuple[str, int]] = Counter()
    trajectory_hashes: set[str] = set()
    longest: list[tuple[int, int, dict[str, Any]]] = []
    identities: set[str] = set()
    for index, (student, target) in enumerate(zip(students, targets, strict=True)):
        identity = str(student.get("index"))
        source = str(student.get("source_dataset"))
        label = int(student.get("label", -1))
        if (
            identity != str(target.get("index"))
            or label != int(target.get("label", -1))
            or student.get("dataset") != target.get("dataset")
        ):
            raise ValueError(f"student/target alignment drift at row {index}")
        if (
            identity in identities
            or source not in EXPECTED_SOURCES
            or label not in {0, 1}
        ):
            raise ValueError(f"training identity/source/label drift at row {index}")
        probability = float(target.get("soft_target", math.nan))
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"invalid soft target at row {index}")
        if not str(student.get("dataset", "")).startswith("tool_trajectory/"):
            raise ValueError(f"non-monitoring row at index {index}")
        trajectory = str(student.get("trajectory_sha256", ""))
        if len(trajectory) != 64 or not student.get("lineage_group"):
            raise ValueError(f"missing trajectory provenance at row {index}")
        identities.add(identity)
        trajectory_hashes.add(trajectory)
        strata[(source, label)] += 1
        candidate = (int(student["student_direct_tokens"]), index, student)
        if len(longest) < 32:
            heapq.heappush(longest, candidate)
        elif candidate[:2] > longest[0][:2]:
            heapq.heapreplace(longest, candidate)
    if set(source for source, _ in strata) != EXPECTED_SOURCES or len(strata) != 10:
        raise ValueError(f"training source-label strata drifted: {strata}")
    selection = [
        {
            "dataset": row["dataset"],
            "index": row["index"],
            "label": row["label"],
            "lineage_group": row["lineage_group"],
            "student_direct_tokens": tokens,
            "trajectory_sha256": row["trajectory_sha256"],
        }
        for tokens, _, row in sorted(longest, reverse=True)
    ]
    return {
        "rows": len(students),
        "unique_ids": len(identities),
        "unique_trajectories": len(trajectory_hashes),
        "trajectory_hashes": trajectory_hashes,
        "source_label_counts": {
            f"{source}:{label}": count
            for (source, label), count in sorted(strata.items())
        },
        "maximum_student_direct_tokens": selection[0]["student_direct_tokens"],
    }, selection


def validate_id_separation(path: Path, training_hashes: set[str]) -> dict[str, Any]:
    if sha256_file(path) != ID_INPUT_SHA256:
        raise ValueError("held-out ID checksum drifted")
    rows = read_jsonl(path)
    ids = [str(row.get("id")) for row in rows]
    hashes = {str(row.get("metadata", {}).get("trajectory_sha256")) for row in rows}
    overlap = hashes & training_hashes
    if len(rows) != 3_012 or len(ids) != len(set(ids)) or overlap:
        raise ValueError(
            f"held-out ID separation failed: rows={len(rows)} overlap={len(overlap)}"
        )
    return {"rows": len(rows), "unique_ids": len(set(ids)), "trajectory_overlap": 0}


def make_jobs(data_dir: Path, result_dir: Path) -> list[dict[str, Any]]:
    steps = math.ceil(TRAIN_ROWS / EFFECTIVE_BATCH_SIZE)
    common = {
        "capacity_kind": "lora",
        "data_scope": "monitoring_only",
        "deception_rows": 0,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "seed": TRAINING_SEED,
        "target": "kimi_soft",
        "rank": RANK,
        "lora_alpha": LORA_ALPHA,
        "max_length": MAX_LENGTH,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "optimizer": "adamw",
        "lr_scheduler_type": "linear",
        "warmup_ratio": 0.03,
        "weight_decay": 0.0,
        "num_train_epochs": 1.0,
        "max_steps": -1,
        "soft_loss_weight": 1.0,
        "direct_loss_weight": 0.0,
        "require_causal_conv1d": True,
        "fla_disable_backend_dispatch": True,
        "triton_version": "3.7.1",
        "train_rows": TRAIN_ROWS,
        "selection_manifest": None,
        "student_rows": (data_dir / "student_rows.jsonl").as_posix(),
        "soft_targets": (data_dir / "soft_targets.jsonl").as_posix(),
        "student_rows_sha256": STUDENT_ROWS_SHA256,
        "soft_targets_sha256": SOFT_TARGETS_SHA256,
        "save_steps": steps,
    }
    jobs = []
    for rate in LEARNING_RATES:
        name = job_name(rate)
        output = result_dir / "runs" / name
        jobs.append(
            {
                **common,
                "job_name": name,
                "design_role": "retrained_control" if rate == 5e-5 else "lr_candidate",
                "learning_rate": rate,
                "output_dir": output.as_posix(),
                "causal_adapter_dir": (output / "causal_adapter").as_posix(),
                "model_dir": (output / "model").as_posix(),
            }
        )
    validate_jobs(jobs)
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--id-input", type=Path, default=DEFAULT_ID_INPUT)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    data_dir = args.data_dir.resolve()
    audit, selection = inspect_training_inputs(
        data_dir / "student_rows.jsonl", data_dir / "soft_targets.jsonl"
    )
    id_audit = validate_id_separation(
        args.id_input.resolve(), audit.pop("trajectory_hashes")
    )
    selection_path = result_dir / "selections" / "preflight-longest-32.jsonl"
    jobs_path = result_dir / "jobs.jsonl"
    atomic_write_jsonl(selection_path, selection)
    jobs = make_jobs(data_dir, result_dir)
    atomic_write_jsonl(jobs_path, jobs)
    manifest = {
        "campaign_id": "gleipnir4b-monitoring-only-lr-sweep-v5",
        "training": audit,
        "held_out_id": id_audit,
        "jobs": [job["job_name"] for job in jobs],
        "jobs_sha256": sha256_file(jobs_path),
        "preflight_selection": selection_path.as_posix(),
        "preflight_selection_sha256": sha256_file(selection_path),
    }
    atomic_write_json(result_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
