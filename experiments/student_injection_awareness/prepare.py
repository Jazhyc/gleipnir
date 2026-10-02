"""Freeze matched student instructions without changing cached teacher targets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from experiments.tool_trajectory_monitoring.prompting import (
    PromptTemplate,
    load_prompt_set,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/student_injection_awareness"
OUTPUT = ROOT / "results/student_injection_awareness"
CONFIG = Path(__file__).with_name("config.yaml")
VARIANTS = ("regular", "injection_aware")


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


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


def trajectory_from_prompt(prompt: str, expected_hash: str) -> str:
    """Recover source bytes by envelope and checksum, including trailing newlines."""
    template = load_prompt_set().student
    suffix = f"{template.trajectory_close}\n"
    if not prompt.startswith(template.cache_prefix) or not prompt.endswith(suffix):
        raise ValueError("source student envelope drift")
    value = prompt[len(template.cache_prefix) : -len(suffix)]
    for candidate in (value, value.removesuffix("\n")):
        if digest(candidate) == expected_hash:
            return candidate
    raise ValueError("trajectory checksum drift")


def rerender_training(
    rows: list[dict[str, Any]], template: PromptTemplate
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        if not row["dataset"].startswith("tool_trajectory/"):
            raise ValueError("deception data in monitoring-only campaign")
        if digest(row["student_prompt"]) != row["student_prompt_sha256"]:
            raise ValueError("source student prompt checksum drift")
        trajectory = trajectory_from_prompt(
            row["student_prompt"], row["trajectory_sha256"]
        )
        prompt = template.render(trajectory)
        rendered = {
            **row,
            "student_prompt": prompt,
            "student_prompt_sha256": digest(prompt),
            "student_template_sha256": template.template_sha256,
            "prompt_set_id": template.prompt_set_id,
        }
        rendered.pop("student_direct_tokens", None)
        output.append(rendered)
    return output


def rerender_evaluation(
    rows: list[dict[str, Any]], template: PromptTemplate
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        metadata = row["metadata"]
        if digest(row["prompt"]) != metadata["rendered_prompt_sha256"]:
            raise ValueError("evaluation source prompt drift")
        trajectory = trajectory_from_prompt(
            row["prompt"], metadata["trajectory_sha256"]
        )
        prompt = template.render(trajectory)
        output.append(
            {
                "id": row["id"],
                "prompt": prompt,
                "metadata": {
                    "source_dataset": metadata["source_dataset"],
                    "ground_truth": metadata["ground_truth"],
                    "trajectory_sha256": digest(trajectory),
                    "prompt_template_sha256": template.template_sha256,
                    "prompt_set_id": template.prompt_set_id,
                    "rendered_prompt_sha256": digest(prompt),
                    "original_metadata": metadata,
                },
            }
        )
    return output


def validate_targets(rows: list[dict[str, Any]], targets: list[dict[str, Any]]) -> None:
    identities = [(r["dataset"], str(r["index"])) for r in rows]
    by_key = {(r["dataset"], str(r["index"])): r for r in targets}
    if (
        len(set(identities)) != len(rows)
        or len(by_key) != len(targets)
        or set(identities) != set(by_key)
    ):
        raise ValueError("teacher identity coverage drift")
    for row in rows:
        target = by_key[(row["dataset"], str(row["index"]))]
        if target["label"] != row["label"] or not 0 <= target["soft_target"] <= 1:
            raise ValueError("teacher label or probability drift")
        if target["rendered_prompt_sha256"] != row["teacher_rendered_prompt_sha256"]:
            raise ValueError("teacher prompt provenance drift")


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
