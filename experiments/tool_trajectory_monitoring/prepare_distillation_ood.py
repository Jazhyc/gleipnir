#!/usr/bin/env python3
"""Materialize the frozen OOD suite under the compact student prompt."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.tool_trajectory_monitoring.prompting import load_prompt_set

DEFAULT_SOURCE = Path(
    "data/tool_trajectory_monitoring/teacher_ood_benchmark/prompts.jsonl"
)


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


DEFAULT_OUTPUT = Path("data/tool_trajectory_monitoring/distillation_ood/prompts.jsonl")


def extract_trajectory(rendered: str, prefix: str, suffix: str) -> str:
    """Recover an unchanged trajectory from a validated rendered prompt."""
    if not rendered.startswith(prefix) or not rendered.endswith(suffix):
        raise ValueError("teacher prompt does not match the frozen trajectory envelope")
    trajectory = rendered[len(prefix) : -len(suffix)]
    if not trajectory.strip():
        raise ValueError("teacher prompt contains an empty trajectory")
    return trajectory


def materialize_rows(
    source_rows: list[dict[str, Any]],
    *,
    expected_rows: int = 6_395,
) -> list[dict[str, Any]]:
    """Rerender exact teacher trajectories with the compact student prompt."""
    prompts = load_prompt_set()
    teacher_suffix = f"{prompts.teacher.trajectory_close}\n"
    output = []
    for source in source_rows:
        metadata = dict(source["metadata"])
        rendered = str(source["prompt"])
        if hashlib.sha256(rendered.encode()).hexdigest() != metadata.get(
            "rendered_prompt_sha256"
        ):
            raise ValueError(f"teacher prompt checksum drift for id={source['id']!r}")
        trajectory = extract_trajectory(
            rendered, prompts.teacher.cache_prefix, teacher_suffix
        )
        expected_trajectory_sha256 = metadata.get("trajectory_sha256")
        if (
            hashlib.sha256(trajectory.encode()).hexdigest()
            != expected_trajectory_sha256
            and trajectory.endswith("\n")
            and hashlib.sha256(trajectory[:-1].encode()).hexdigest()
            == expected_trajectory_sha256
        ):
            trajectory = trajectory[:-1]
        if (
            hashlib.sha256(trajectory.encode()).hexdigest()
            != expected_trajectory_sha256
        ):
            raise ValueError(f"trajectory checksum drift for id={source['id']!r}")
        student_prompt = prompts.student.render(trajectory)
        student_sha256 = hashlib.sha256(student_prompt.encode()).hexdigest()
        metadata.update(
            prompt_set_id=prompts.prompt_set_id,
            prompt_role="student",
            prompt_template_sha256=prompts.student.template_sha256,
            teacher_rendered_prompt_sha256=metadata["rendered_prompt_sha256"],
            rendered_prompt_sha256=student_sha256,
        )
        output.append(
            {
                "id": str(source["id"]),
                "metadata": metadata,
                "prompt": student_prompt,
            }
        )
    identities = [row["id"] for row in output]
    if len(output) != expected_rows or len(identities) != len(set(identities)):
        raise ValueError(
            f"student evaluation suite must contain {expected_rows:,} unique rows"
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_rows = read_jsonl(args.source)
    rows = materialize_rows(source_rows)
    atomic_write_jsonl(args.output, rows)
    prompts = load_prompt_set()
    counts = Counter(
        (row["metadata"]["source_dataset"], row["metadata"]["ground_truth"])
        for row in rows
    )
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    atomic_write_json(
        manifest_path,
        {
            "campaign_id": "tool-trajectory-distilled-student-ood-v1",
            "source": args.source.as_posix(),
            "source_sha256": sha256_file(args.source),
            "output": args.output.as_posix(),
            "output_sha256": sha256_file(args.output),
            "rows": len(rows),
            "source_label_counts": {
                f"{source}:{label}": count
                for (source, label), count in sorted(counts.items())
            },
            "prompt_set_id": prompts.prompt_set_id,
            "teacher_template_sha256": prompts.teacher.template_sha256,
            "student_template_sha256": prompts.student.template_sha256,
            "transformation": (
                "Validate and extract the exact trajectory from each frozen teacher "
                "prompt, then rerender it under the compact student prompt without "
                "changing row membership, labels, or trajectory text."
            ),
        },
    )
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
