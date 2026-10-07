"""Freeze unchanged A/B inputs, cached baseline and fixed adapter identities."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from gleipnir.evaluation.sources import (
    BINARY_SOURCE_PATHS,
    DECISION_SOURCE_PATHS,
    PREFERENCE_SOURCE_PATHS,
    SCORING_SOURCE_PATHS,
)
from gleipnir.monitoring_campaign_data import file_hash, read_rows, write_json
from gleipnir.monitoring_scoring import completed_predictions

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CONFIG = HERE / "config.yaml"
DATA = ROOT / "data/augmented_judge_evaluation"
OUTPUT = ROOT / "results/augmented_judge_evaluation"
SOURCE = ROOT / "data/judge_injection_continuation"
BASELINE = ROOT / "results/judge_injection_continuation/evaluation_regular"


def configuration() -> dict:
    return yaml.safe_load(CONFIG.read_text())


def verify() -> dict:
    manifest = json.loads((DATA / "manifest.json").read_text())
    for path, expected in manifest["files_sha256"].items():
        if file_hash(ROOT / path) != expected:
            raise ValueError(f"frozen evaluation drift: {path}")
    return manifest


def main() -> None:
    if (DATA / "manifest.json").exists():
        verify()
        print("preparation_reused", flush=True)
        return
    source = json.loads((SOURCE / "manifest.json").read_text())
    for name in ("test.jsonl", "canary.jsonl", "train.jsonl"):
        if file_hash(SOURCE / name) != source["files_sha256"][name]:
            raise ValueError("original auxiliary source drift")
    test, canary, train = (
        read_rows(SOURCE / n) for n in ("test.jsonl", "canary.jsonl", "train.jsonl")
    )
    groups = {r["lineage_group"] for r in test}
    if (
        len(test) != 4188
        or len(groups) != 6
        or groups & {r["lineage_group"] for r in train + canary}
    ):
        raise ValueError("original query holdout drift or leakage")
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    template = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True
    ).from_string((SOURCE / "tokenizer/chat_template.jinja").read_text())
    ident = json.loads((BASELINE / "identity.json").read_text())
    original = completed_predictions(
        BASELINE / "original/preferences.jsonl", test, ident
    )
    hashes = {r["id"]: r["prompt_sha256"] for r in original}
    from gleipnir.monitoring_campaign_data import digest

    if any(
        hashes[r["id"]]
        != digest(
            template.render(
                messages=[{"role": "user", "content": r["student_prompt"]}],
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
        for r in test
    ):
        raise ValueError("cached original prompt drift")
    DATA.mkdir(parents=True, exist_ok=True)
    for name in ("test.jsonl", "canary.jsonl"):
        shutil.copyfile(SOURCE / name, DATA / name)
    files = [CONFIG, DATA / "test.jsonl", DATA / "canary.jsonl"]
    files += [
        BASELINE / "original" / ("preferences" + suffix)
        for suffix in (".jsonl", ".contract.json", ".complete.json")
    ]
    files += [BASELINE / "identity.json"]
    adapters = {}
    for size, model in configuration()["models"].items():
        adapters[size] = {}
        for name, relative in model["adapters"].items():
            folder = ROOT / relative
            paths = {
                layout: folder / directory
                for layout, directory in (
                    ("master", "causal_adapter"),
                    ("serving", "model"),
                )
            }
            weights = {
                k: file_hash(p / "adapter_model.safetensors") for k, p in paths.items()
            }
            if name == "augmented":
                receipt = json.loads((folder / "complete.json").read_text())
                if (
                    receipt["steps"] != 272
                    or receipt["status"] != "trained"
                    or any(weights[k] != receipt[k + "_sha256"] for k in weights)
                ):
                    raise ValueError("augmented training/checkpoint drift")
                files.append(folder / "complete.json")
            rebase = json.loads((paths["serving"] / "rebase_manifest.json").read_text())
            if weights != {
                "master": rebase["source_sha256"],
                "serving": rebase["destination_sha256"],
            }:
                raise ValueError("master/serving export identity drift")
            for path in paths.values():
                files += [
                    path / "adapter_model.safetensors",
                    path / "adapter_config.json",
                ]
            files.append(paths["serving"] / "rebase_manifest.json")
            adapters[size][name] = weights
    regular = ROOT / "results/student_injection_awareness/4b/regular"
    if any(
        file_hash(regular / directory / "adapter_model.safetensors")
        != ident["adapters"]["original"][layout]
        for layout, directory in (("master", "causal_adapter"), ("serving", "model"))
    ):
        raise ValueError("cached original adapter drift")
    files += sorted((HERE).glob("*.py"))
    files += [
        ROOT / p
        for p in (
            *SCORING_SOURCE_PATHS,
            *PREFERENCE_SOURCE_PATHS,
            *DECISION_SOURCE_PATHS,
            *BINARY_SOURCE_PATHS,
            "src/gleipnir/data/monitoring.py",
            "uv.lock",
            "experiments/training_procedure_screen/evaluate_causal.py",
            "experiments/deception_distillation/train_student_sft.py",
        )
    ]
    write_json(
        DATA / "manifest.json",
        {
            "campaign_id": configuration()["campaign_id"],
            "surface": "AB",
            "rows": len(test),
            "query_groups": len(groups),
            "test_fraction": source["actual_test_fraction"],
            "original_manifest_sha256": file_hash(SOURCE / "manifest.json"),
            "adapters": adapters,
            "cached_regular_4b_sha256": ident["adapters"]["original"],
            "cached_regular_4b_rescored": False,
            "files_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in files},
        },
    )
    print("preparation_complete 4188 rows 6 query groups", flush=True)


if __name__ == "__main__":
    main()
