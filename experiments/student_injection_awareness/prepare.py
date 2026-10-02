"""Freeze matched student instructions without changing cached teacher targets."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import yaml

from experiments.tool_trajectory_monitoring.prompting import (
    PromptTemplate,
    load_prompt_set,
)
from gleipnir.monitoring_campaign_data import digest as digest
from gleipnir.monitoring_campaign_data import (
    file_hash,
    read_rows,
    rerender_evaluation,
    rerender_training,
    validate_targets,
    write_json,
    write_rows,
)
from gleipnir.monitoring_campaign_data import (
    trajectory_from_prompt as trajectory_from_prompt,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/student_injection_awareness"
OUTPUT = ROOT / "results/student_injection_awareness"
CONFIG = Path(__file__).with_name("config.yaml")
VARIANTS = ("regular", "injection_aware")


def templates() -> dict[str, PromptTemplate]:
    regular = load_prompt_set().student
    instruction = (
        ROOT
        / "experiments/tool_trajectory_monitoring/prompts"
        / "student_prompt_injection_aware.txt"
    ).read_text()
    return {
        "regular": regular,
        "injection_aware": replace(
            regular,
            instruction=instruction,
            prompt_set_id="tool-trajectory-student-injection-aware-v1",
        ),
    }


def main() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    sources = {
        key: ROOT / config[key]
        for key in [
            "training_source",
            "teacher_source",
            "id_source",
            "id_manifest",
            "ood_source",
        ]
    }
    hashes = {key: file_hash(path) for key, path in sources.items()}
    for key in ["id_source", "id_manifest", "ood_source"]:
        if hashes[key] != config[f"{key}_sha256"]:
            raise ValueError(f"frozen source drift: {key}")
    rows = read_rows(sources["training_source"])
    targets = read_rows(sources["teacher_source"])
    if len(rows) != config["training_rows"]:
        raise ValueError("training population drift")
    validate_targets(rows, targets)
    if (DATA / "manifest.json").exists():
        previous = json.loads((DATA / "manifest.json").read_text())
        if previous["sources_sha256"] != hashes or previous[
            "config_sha256"
        ] != file_hash(CONFIG):
            raise ValueError("prepared campaign source/config drift")
        for relative, expected in previous["files_sha256"].items():
            if file_hash(DATA / relative) != expected:
                raise ValueError(f"prepared artifact drift: {relative}")
        if {v: t.template_sha256 for v, t in templates().items()} != previous[
            "template_sha256"
        ]:
            raise ValueError("prepared instruction drift")
        print("Reusing verified preparation", flush=True)
        return
    write_rows(DATA / "soft_targets.jsonl", targets)
    files = ["soft_targets.jsonl"]
    for variant, template in templates().items():
        write_rows(
            DATA / variant / "student_rows.jsonl", rerender_training(rows, template)
        )
        files.append(f"{variant}/student_rows.jsonl")
        for split in ["id", "ood"]:
            population = read_rows(sources[f"{split}_source"])
            expected = 3012 if split == "id" else 6395
            if (
                len(population) != expected
                or len({r["id"] for r in population}) != expected
            ):
                raise ValueError(f"{split} population drift")
            write_rows(
                DATA / variant / f"{split}.jsonl",
                rerender_evaluation(population, template),
            )
            files.append(f"{variant}/{split}.jsonl")
    write_json(
        DATA / "manifest.json",
        {
            "campaign_id": config["campaign_id"],
            "config_sha256": file_hash(CONFIG),
            "sources_sha256": hashes,
            "template_sha256": {v: t.template_sha256 for v, t in templates().items()},
            "files_sha256": {p: file_hash(DATA / p) for p in files},
            "training_rows": len(rows),
            "teacher_targets_unchanged": True,
            "selection": "final one-epoch checkpoint; no ID/OOD selection",
        },
    )
    print("Prepared matched training and ID/OOD variants", flush=True)


if __name__ == "__main__":
    main()
