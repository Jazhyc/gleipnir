"""Hashed monitoring-campaign data with separate source labels and teacher targets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.tool_trajectory_monitoring.prompting import (
    PromptTemplate,
    load_prompt_set,
)


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


def trajectory_from_prompt(
    prompt: str, expected_hash: str, *, source_template: PromptTemplate | None = None
) -> str:
    """Recover source bytes by envelope and checksum, including trailing newlines."""
    template = (
        source_template if source_template is not None else load_prompt_set().student
    )
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
    source_template = load_prompt_set().student
    for row in rows:
        if not row["dataset"].startswith("tool_trajectory/"):
            raise ValueError("deception data in monitoring-only campaign")
        if digest(row["student_prompt"]) != row["student_prompt_sha256"]:
            raise ValueError("source student prompt checksum drift")
        trajectory = trajectory_from_prompt(
            row["student_prompt"],
            row["trajectory_sha256"],
            source_template=source_template,
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
    source_template = load_prompt_set().student
    for row in rows:
        metadata = row["metadata"]
        if digest(row["prompt"]) != metadata["rendered_prompt_sha256"]:
            raise ValueError("evaluation source prompt drift")
        trajectory = trajectory_from_prompt(
            row["prompt"],
            metadata["trajectory_sha256"],
            source_template=source_template,
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
