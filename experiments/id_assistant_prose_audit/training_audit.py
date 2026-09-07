"""Audit reasoning actually present in monitoring teacher and student inputs."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from experiments.id_assistant_prose_audit.run import HEADER, digest
from experiments.tool_trajectory_monitoring.prepare_distillation_ood import (
    atomic_write_json,
    extract_trajectory,
    sha256_file,
)
from experiments.tool_trajectory_monitoring.prompting import load_prompt_set


def typed_reasoning(messages: list[dict[str, Any]]) -> list[str]:
    """Return nonempty, explicitly typed assistant reasoning, excluding prose."""
    texts = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if block.get("type") == "reasoning" and block.get("reasoning", "").strip():
                texts.append(block["reasoning"].strip())
    return texts


def count_think_calls(trajectory: str) -> int:
    """Count think-call starts only inside serialized tool-call blocks."""
    import re

    matches = list(HEADER.finditer(trajectory))
    return sum(
        len(
            re.findall(
                r"^think\(",
                trajectory[
                    m.end() : matches[i + 1].start()
                    if i + 1 < len(matches)
                    else len(trajectory)
                ],
                re.M,
            )
        )
        for i, m in enumerate(matches)
        if m[1].lower() == "tool calls"
    )


def main() -> None:
    config = json.loads(Path(__file__).with_name("training_config.json").read_text())
    for path, expected in config["files"].items():
        if sha256_file(Path(path)) != expected:
            raise ValueError(f"Frozen audit source drift: {path}")
    templates = load_prompt_set()
    raw_by_hash = {}
    for path in config["raw_stride_files"]:
        for row in pq.read_table(
            path, columns=["id", "prompt", "trajectory_data"]
        ).to_pylist():
            text = row["prompt"][0]["content"]
            key = digest(text)
            if key in raw_by_hash:
                raise ValueError("Ambiguous STRIDE source trajectory")
            raw_by_hash[key] = row
    prompts, trajectories = {}, {}
    counts = defaultdict(Counter)
    audit_rows = []
    for line in Path(config["teacher_prompts"]).open():
        row = json.loads(line)
        metadata = row["metadata"]
        if digest(row["prompt"]) != metadata["rendered_prompt_sha256"]:
            raise ValueError("Teacher prompt hash drift")
        trajectory = extract_trajectory(
            row["prompt"],
            templates.teacher.cache_prefix,
            f"{templates.teacher.trajectory_close}\n",
        )
        if digest(trajectory) != metadata["trajectory_sha256"]:
            trajectory = trajectory.removesuffix("\n")
        if digest(trajectory) != metadata["trajectory_sha256"]:
            raise ValueError("Teacher trajectory hash drift")
        source = metadata["source_dataset"]
        n_calls = count_think_calls(trajectory)
        reasoning, source_id = [], None
        if source == "stride":
            raw = raw_by_hash[digest(trajectory)]
            source_id = raw["id"]
            messages = json.loads(raw["trajectory_data"])["messages"]
            reasoning = typed_reasoning(messages)
            if any(text not in trajectory for text in reasoning):
                raise ValueError("A typed reasoning block is not in the actual input")
            raw_calls = sum(
                c.get("function") == "think"
                for m in messages
                for c in m.get("tool_calls") or []
            )
            if n_calls != raw_calls:
                raise ValueError("Rendered and structured think-call counts differ")
        stats = {
            "rows": 1,
            "explicit_think_calls": n_calls,
            "rows_with_think": int(n_calls > 0),
            "verified_typed_reasoning_blocks": len(reasoning),
            "rows_with_verified_typed_reasoning": int(bool(reasoning)),
            "rows_with_verified_thinking": int(bool(n_calls or reasoning)),
        }
        counts[source].update(stats)
        counts["total"].update(stats)
        prompts[row["id"]] = row
        trajectories[row["id"]] = trajectory
        audit_rows.append(
            {"id": row["id"], "source": source, "raw_stride_id": source_id, **stats}
        )
    cached = set()
    for path in config["teacher_caches"]:
        for line in Path(path).open():
            row = json.loads(line)
            if row["id"] in cached or row["prompt_sha256"] != digest(
                prompts[row["id"]]["prompt"]
            ):
                raise ValueError("Teacher cache identity or prompt drift")
            cached.add(row["id"])
    if cached != set(prompts) or len(prompts) != 8688:
        raise ValueError("Teacher cache coverage drift")
    matched = set()
    for line in Path(config["mixed_student_rows"]).open():
        row = json.loads(line)
        if row["index"] in prompts:
            expected = templates.student.render(trajectories[row["index"]])
            if row["student_prompt"] != expected or row["index"] in matched:
                raise ValueError(
                    "Actual mixed student input differs from teacher trajectory"
                )
            matched.add(row["index"])
    if matched != set(prompts):
        raise ValueError("Mixed student coverage drift")
    result = {
        "config": config,
        "counts": dict(counts),
        "teacher_cache_exact_prompt_matches": len(cached),
        "mixed_student_exact_trajectory_matches": len(matched),
        "interpretation": (
            "Typed reasoning verified against raw STRIDE; other sources only "
            "scanned for explicit think calls. Prior deception rows excluded."
        ),
    }
    output = Path("results/training_reasoning_audit")
    atomic_write_json(output / "summary.json", result)
    atomic_write_json(output / "rows.json", audit_rows)
    print(json.dumps(result["counts"], indent=2))


if __name__ == "__main__":
    main()
