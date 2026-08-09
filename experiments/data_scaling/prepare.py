#!/usr/bin/env python3
"""Audit inputs, build nested train manifests, and materialize validation prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.data_scaling.core import (  # noqa: E402
    nested_stratified_selections,
    record_identity,
)
from gleipnir.prompts import build_student_prompt  # noqa: E402

DEFAULT_FRACTIONS = (0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0)
DEFAULT_SEEDS = (0, 1, 2)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from error
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def source_prompt_template(records: list[dict[str, Any]]) -> str:
    templates = {
        str(record["student_prompt"]).partition("<context>")[0].rstrip()
        for record in records
        if str(record["dataset"]).startswith("aletheias-quest/")
    }
    if len(templates) != 1:
        raise ValueError(
            f"expected one competition student prompt template, found {len(templates)}"
        )
    return templates.pop()


def load_validation(
    split_yaml: Path,
    prompt_template: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from datasets import load_dataset

    specification = yaml.safe_load(split_yaml.read_text())
    split_root = split_yaml.parent.parent
    rows: list[dict[str, Any]] = []
    sources = []
    for dataset_config in specification["datasets"]:
        name = str(dataset_config["name"])
        labels_path = split_root / str(dataset_config["labels_uri"])
        labels = pd.read_csv(labels_path)
        id_column = str(dataset_config.get("id_column", "index"))
        label_column = str(dataset_config.get("label_column", "deceptive"))
        label_by_index = {
            index: int(label)
            for index, label in zip(
                labels[id_column], labels[label_column], strict=True
            )
        }
        dataset = load_dataset(name, split="test")
        if "index" not in dataset.column_names:
            dataset = dataset.add_column("index", list(range(len(dataset))))
        found = set()
        for source_row in dataset:
            index = source_row["index"]
            if index not in label_by_index:
                continue
            prompt = build_student_prompt(
                source_row["messages"], prompt_template, 3000, "tail"
            )
            rows.append(
                {
                    "dataset": name,
                    "index": index,
                    "label": label_by_index[index],
                    "student_prompt": prompt,
                    "student_prompt_sha256": hashlib.sha256(
                        prompt.encode("utf-8")
                    ).hexdigest(),
                }
            )
            found.add(index)
        if found != set(label_by_index):
            missing = sorted(set(label_by_index) - found)
            raise RuntimeError(f"{name}: missing {len(missing)} labeled rows")
        sources.append(
            {
                "dataset": name,
                "dataset_fingerprint": dataset._fingerprint,
                "labels": labels_path.as_posix(),
                "labels_sha256": sha256_file(labels_path),
                "rows": len(labels),
            }
        )
    return rows, {"split": split_yaml.as_posix(), "sources": sources}


def fraction_tag(fraction: float) -> str:
    return f"f{round(fraction * 100):03d}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-rows", type=Path, required=True)
    parser.add_argument("--soft-targets", type=Path, required=True)
    parser.add_argument("--validation-split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/data_scaling"))
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=Path("data/data_scaling/validation.jsonl"),
    )
    parser.add_argument("--fractions", nargs="+", type=float, default=DEFAULT_FRACTIONS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    args = parser.parse_args()

    student_path = args.student_rows.resolve()
    soft_path = args.soft_targets.resolve()
    split_path = args.validation_split.resolve()
    output_dir = args.output_dir.resolve()
    validation_output = args.validation_output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(student_path)
    targets = read_jsonl(soft_path)
    if len(records) != 13149 or len(targets) != len(records):
        raise ValueError(
            "expected matching 13,149-row artifacts, got "
            f"{len(records)} and {len(targets)}"
        )
    target_by_identity = {record_identity(record): record for record in targets}
    if len(target_by_identity) != len(targets):
        raise ValueError("soft-target artifact contains duplicate identities")
    for record in records:
        identity = record_identity(record)
        target = target_by_identity.get(identity)
        if target is None or int(target["label"]) != int(record["label"]):
            raise ValueError(f"missing or label-inconsistent target for {identity!r}")

    prompt_template = source_prompt_template(records)
    validation, validation_provenance = load_validation(split_path, prompt_template)
    train_identities = {record_identity(record) for record in records}
    overlap = train_identities & {record_identity(record) for record in validation}
    if overlap:
        raise ValueError(f"training/validation identity leakage: {sorted(overlap)[:3]}")
    write_jsonl(validation_output, validation)

    jobs = []
    selection_summary: dict[str, Any] = {}
    for seed in sorted(set(args.seeds)):
        selections = nested_stratified_selections(records, args.fractions, seed)
        seed_summary = {}
        for fraction, selected in selections.items():
            tag = fraction_tag(fraction)
            selection_path = output_dir / "selections" / f"seed{seed}-{tag}.jsonl"
            manifest_rows = [
                {
                    "dataset": record["dataset"],
                    "index": record["index"],
                    "label": int(record["label"]),
                }
                for record in selected
            ]
            write_jsonl(selection_path, manifest_rows)
            counts = Counter(
                (str(record["dataset"]), int(record["label"]))
                for record in selected
            )
            job_name = f"seed{seed}-{tag}"
            job_dir = output_dir / "runs" / job_name
            jobs.append(
                {
                    "job_name": job_name,
                    "seed": seed,
                    "fraction": fraction,
                    "train_rows": len(selected),
                    "selection_manifest": selection_path.as_posix(),
                    "selection_sha256": sha256_file(selection_path),
                    "student_rows": student_path.as_posix(),
                    "soft_targets": soft_path.as_posix(),
                    "output_dir": job_dir.as_posix(),
                    "adapter_dir": (job_dir / "adapter").as_posix(),
                }
            )
            seed_summary[tag] = {
                "fraction": fraction,
                "rows": len(selected),
                "counts": {
                    f"{dataset}|{label}": count
                    for (dataset, label), count in sorted(counts.items())
                },
            }
        selection_summary[str(seed)] = seed_summary
    jobs_path = output_dir / "jobs.jsonl"
    write_jsonl(jobs_path, jobs)

    manifest = {
        "hypothesis": (
            "fixed-recipe validation error decreases as a bounded power law in "
            "the number of Kimi K3 annotated training examples"
        ),
        "student_rows": student_path.as_posix(),
        "student_rows_sha256": sha256_file(student_path),
        "soft_targets": soft_path.as_posix(),
        "soft_targets_sha256": sha256_file(soft_path),
        "training_rows": len(records),
        "training_source_counts": dict(
            sorted(Counter(str(record["dataset"]) for record in records).items())
        ),
        "fractions": sorted(set(args.fractions)),
        "seeds": sorted(set(args.seeds)),
        "jobs": len(jobs),
        "jobs_path": jobs_path.as_posix(),
        "validation_output": validation_output.as_posix(),
        "validation_sha256": sha256_file(validation_output),
        "validation_rows": len(validation),
        "validation_datasets": len({record["dataset"] for record in validation}),
        "validation_provenance": validation_provenance,
        "competition_prompt_template_sha256": hashlib.sha256(
            prompt_template.encode("utf-8")
        ).hexdigest(),
        "train_validation_overlap": 0,
        "selection": selection_summary,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"prepared {len(jobs)} jobs, {len(records)} training rows, and "
        f"{len(validation)} fixed validation rows; manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()
