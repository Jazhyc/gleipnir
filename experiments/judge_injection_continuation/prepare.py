"""Freeze entire-source provenance, grouped split and one-token preference inputs."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter
from pathlib import Path

import yaml

from gleipnir.judge_injection import augment_pairs, digest, grouped_split, load_pairs
from gleipnir.monitoring_campaign_data import (
    file_hash,
    read_rows,
    write_json,
    write_rows,
)

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = ROOT / "data/judge_injection_continuation"
OUTPUT = ROOT / "results/judge_injection_continuation"
CONFIG = HERE / "config.yaml"


def configuration() -> dict:
    return yaml.safe_load(CONFIG.read_text())


def verify_preparation() -> dict:
    manifest = json.loads((DATA / "manifest.json").read_text())
    if manifest["config_sha256"] != file_hash(CONFIG) or manifest[
        "prompt_sha256"
    ] != file_hash(HERE / "prompt.txt"):
        raise ValueError("frozen configuration or prompt drift")
    for path, expected in manifest["files_sha256"].items():
        if file_hash(DATA / path) != expected:
            raise ValueError(f"data artifact drift: {path}")
    for path, expected in manifest["source_artifact_sha256"].items():
        if file_hash(ROOT / path) != expected:
            raise ValueError(f"source adapter/provenance drift: {path}")
    train = read_rows(DATA / "train.jsonl")
    test = read_rows(DATA / "test.jsonl")
    if {r["lineage_group"] for r in train} & {r["lineage_group"] for r in test}:
        raise ValueError("grouped leakage")
    if len(train) != manifest["training_rows"] or len(test) != manifest["test_rows"]:
        raise ValueError("prepared row coverage drift")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    if (DATA / "manifest.json").exists():
        manifest = verify_preparation()
        print(
            f"Reused preparation {manifest['training_rows']} train, "
            f"{manifest['test_rows']} test",
            flush=True,
        )
        return
    config = configuration()
    DATA.mkdir(parents=True, exist_ok=True)
    if args.source:
        shutil.copytree(
            args.source / "dataset", DATA / "raw/dataset", dirs_exist_ok=True
        )
        shutil.copyfile(
            args.source / "download_manifest.json", DATA / "raw/source_manifest.json"
        )
        shutil.copytree(
            args.source / "tokenizer", DATA / "tokenizer", dirs_exist_ok=True
        )
    raw = DATA / "raw"
    source_manifest = json.loads((raw / "source_manifest.json").read_text())
    if source_manifest["revision"] != config["revision"]:
        raise ValueError("upstream revision drift")
    raw_checksums = {}
    for entry in source_manifest["files"]:
        if not entry["path"].startswith("dataset/"):
            continue
        path = raw / entry["path"]
        if file_hash(path) != entry["sha256"]:
            raise ValueError("upstream artifact checksum drift")
        raw_checksums[str(path.relative_to(DATA))] = entry["sha256"]
    pairs, source_counts = load_pairs(raw)
    split = grouped_split(pairs, config["seed"])
    rows = augment_pairs(pairs, split, (HERE / "prompt.txt").read_text())
    write_rows(DATA / "pairs.jsonl", pairs)
    write_json(DATA / "split.json", split)
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    write_rows(DATA / "train.jsonl", train)
    write_rows(DATA / "test.jsonl", test)
    # A mechanical parity cohort uses training inputs, never held-out queries.
    canary = []
    for source in sorted(source_counts):
        for label in (0, 1):
            for condition in ("clean", "preferred_injected", "disfavored_injected"):
                candidates = [
                    r
                    for r in train
                    if (r["source"], r["label"], r["condition"])
                    == (source, label, condition)
                ]
                canary.append(min(candidates, key=lambda r: digest(r["id"])))
    write_rows(DATA / "canary.jsonl", canary)
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(DATA / "tokenizer/tokenizer.json"))
    template = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True
    ).from_string((DATA / "tokenizer/chat_template.jinja").read_text())
    token_counts = {}
    for side, population in (("train", train), ("test", test)):
        lengths = []
        for row in population:
            text = template.render(
                messages=[{"role": "user", "content": row["student_prompt"]}],
                add_generation_prompt=True,
                enable_thinking=False,
            )
            lengths.append(len(tokenizer.encode(text, add_special_tokens=False).ids))
        if max(lengths) >= 29696:
            raise ValueError("training context overflow")
        token_counts[side] = {
            "rows": len(lengths),
            "total_input_tokens": sum(lengths),
            "total_with_decisions": sum(lengths) + len(lengths),
            "maximum": max(lengths),
            "truncated": 0,
        }
    sources = {}
    for relative in (
        config["model"]["initial_adapter"] + "/adapter_model.safetensors",
        config["model"]["initial_adapter"] + "/adapter_config.json",
        config["model"]["original_serving"] + "/adapter_model.safetensors",
        config["model"]["original_serving"] + "/rebase_manifest.json",
        config["startup_validation_reference"],
    ):
        sources[relative] = file_hash(ROOT / relative)
    if (
        sources[config["startup_validation_reference"]]
        != config["startup_validation_reference_sha256"]
    ):
        raise ValueError("startup validation reference drift")
    paths = [
        "pairs.jsonl",
        "split.json",
        "train.jsonl",
        "test.jsonl",
        "canary.jsonl",
        "raw/source_manifest.json",
    ]
    paths.extend(str(p.relative_to(DATA)) for p in (DATA / "tokenizer").iterdir())
    manifest = {
        "campaign_id": config["campaign_id"],
        "config_sha256": file_hash(CONFIG),
        "prompt_sha256": file_hash(HERE / "prompt.txt"),
        "source_revision": config["revision"],
        "source_license": "not declared in upstream repository root; unresolved",
        "source_counts": source_counts,
        "base_pairs": len(pairs),
        "group_counts": dict(Counter(split.values())),
        "training_rows": len(train),
        "test_rows": len(test),
        "actual_test_fraction": len(test) / len(rows),
        "expected_steps": math.ceil(len(train) / 32),
        "row_counts": {
            side: dict(Counter(r["source"] for r in population))
            for side, population in (("train", train), ("test", test))
        },
        "tokens": token_counts,
        "source_artifact_sha256": sources,
        "files_sha256": {**raw_checksums, **{p: file_hash(DATA / p) for p in paths}},
    }
    write_json(DATA / "manifest.json", manifest)
    write_json(OUTPUT / "preparation.json", manifest)
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in (
                    "base_pairs",
                    "group_counts",
                    "training_rows",
                    "test_rows",
                    "actual_test_fraction",
                    "expected_steps",
                    "tokens",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
