"""Freeze the user-selected census filter and unchanged retained supervision."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import yaml

from experiments.monitoring_hard_labels.prepare import validate_holdout
from experiments.monitoring_injection_disentanglement import prepare as previous
from gleipnir.monitoring_campaign_data import (
    digest,
    file_hash,
    read_rows,
    rerender_training,
    write_json,
    write_rows,
)
from gleipnir.monitoring_exclusions import census_filter

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CONFIG = HERE / "config.yaml"
DATA = ROOT / "data/monitoring_injection_removal"
OUTPUT = ROOT / "results/monitoring_injection_removal"
EVAL_DATA = previous.DATA
PROMPTS = previous.PROMPTS


def configuration() -> dict:
    return yaml.safe_load(CONFIG.read_text())


def resolved_profile() -> dict:
    if configuration()["profile"] != previous.configuration()["profile"]:
        raise ValueError("ablation recipe differs from previous campaign")
    return previous.resolved_profile()


def verify_preparation() -> dict:
    manifest = json.loads((DATA / "manifest.json").read_text())
    if manifest["config_sha256"] != file_hash(CONFIG):
        raise ValueError("filtered configuration drift")
    if manifest["profile_sha256"] != digest(
        json.dumps(resolved_profile(), sort_keys=True)
    ):
        raise ValueError("filtered recipe drift")
    if (
        manifest["training_template_sha256"]
        != previous.templates()["neutral"].template_sha256
    ):
        raise ValueError("standard training prompt drift")
    if (HERE / "student_prompt.txt").read_text() != previous.templates()[
        "neutral"
    ].instruction:
        raise ValueError("standard instruction drift")
    previous.verify_preparation()
    for path, expected in manifest["source_sha256"].items():
        if file_hash(ROOT / path) != expected:
            raise ValueError(f"filtered source drift: {path}")
    for path, expected in manifest["files_sha256"].items():
        if file_hash(DATA / path) != expected:
            raise ValueError(f"filtered prepared file drift: {path}")
    return manifest


def main() -> None:
    if (DATA / "manifest.json").exists():
        verify_preparation()
        print("Reusing verified filtered preparation", flush=True)
        return
    config = configuration()
    previous.verify_preparation()
    sources = {}
    for key in (
        "training_source",
        "teacher_source",
        "census_predictions",
        "census_input_audit",
        "previous_manifest",
        "previous_summary",
        "startup_validation_reference",
    ):
        path = ROOT / config[key]
        if file_hash(path) != config[key + "_sha256"]:
            raise ValueError(f"frozen filtering source drift: {key}")
        sources[config[key]] = config[key + "_sha256"]
    audit = json.loads((ROOT / config["census_input_audit"]).read_text())
    if (
        audit["input_sha256"] != config["training_source_sha256"]
        or audit["contract_sha256"] != config["census_contract_sha256"]
    ):
        raise ValueError("census input contract drift")
    training = read_rows(ROOT / config["training_source"])
    targets = read_rows(ROOT / config["teacher_source"])
    retained, retained_targets, excluded = census_filter(
        training,
        targets,
        read_rows(ROOT / config["census_predictions"]),
        threshold=config["filter_threshold"],
        contract_sha256=config["census_contract_sha256"],
    )
    if len(training) != 8688 or len(excluded) != 1154 or len(retained) != 7534:
        raise ValueError("frozen filtering count drift")
    validate_holdout(retained, read_rows(ROOT / config["id_source"]))
    standard = previous.templates()["neutral"]
    if (HERE / "student_prompt.txt").read_text() != standard.instruction:
        raise ValueError("standard instruction differs from original")
    write_rows(DATA / "train/student_rows.jsonl", rerender_training(retained, standard))
    # Keep each original teacher record byte-identical, including provider metadata.
    retained_keys = {(r["dataset"], str(r["index"])) for r in retained_targets}
    target = DATA / "soft_targets.jsonl"
    with (
        (ROOT / config["teacher_source"]).open("rb") as source,
        target.open("wb") as handle,
    ):
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if (row["dataset"], str(row["index"])) in retained_keys:
                handle.write(line)

    def counts(rows: list[dict]) -> dict:
        return dict(
            sorted(Counter(f"{r['source_dataset']}:{r['label']}" for r in rows).items())
        )

    write_json(
        DATA / "selection.json",
        {
            "threshold": config["filter_threshold"],
            "comparison": "score >= threshold",
            "removed": excluded,
            "removed_rows": len(excluded),
            "retained_rows": len(retained),
            "before_source_label_counts": counts(training),
            "retained_source_label_counts": counts(retained),
            "census_predictions_sha256": config["census_predictions_sha256"],
            "limitation": (
                "model-flag removal; false positives and false negatives remain"
            ),
        },
    )
    files = ("train/student_rows.jsonl", "soft_targets.jsonl", "selection.json")
    write_json(
        DATA / "manifest.json",
        {
            "campaign_id": config["campaign_id"],
            "config_sha256": file_hash(CONFIG),
            "profile_sha256": digest(json.dumps(resolved_profile(), sort_keys=True)),
            "training_template_sha256": standard.template_sha256,
            "source_sha256": sources,
            "files_sha256": {p: file_hash(DATA / p) for p in files},
            "training_rows": len(retained),
            "removed_rows": len(excluded),
            "retained_teacher_records_unchanged": True,
            "expected_steps": 236,
            "evaluation": (
                "same fixed injection grid; ID only neutral training instruction"
            ),
        },
    )
    verify_preparation()
    print("Prepared 7534 training rows; excluded 1154 census flags", flush=True)


if __name__ == "__main__":
    main()
