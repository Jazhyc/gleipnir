"""Audit exact frozen OOD inputs and source reasoning-field exclusion."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from experiments.id_assistant_prose_audit.run import digest
from experiments.tool_trajectory_monitoring.prepare_distillation_ood import (
    atomic_write_json,
    atomic_write_jsonl,
    read_jsonl,
    sha256_file,
)
from experiments.tool_trajectory_monitoring.prepare_teacher_ood_cache import (
    OOD_SOURCES,
    SOURCE_DIRECTORY,
    load_source_rows,
)
from experiments.tool_trajectory_monitoring.prompting import load_prompt_set

HEADERS = re.compile(
    r"^(?:\[(USER|SYSTEM|ASSISTANT|TOOL|TOOL CALLS)\][ \t]*\n"
    r"|(USER|SYSTEM|ASSISTANT|TOOL):)",
    re.M,
)
MARKERS = {
    "thinking_tag": r"<\s*/?\s*(?:think(?:ing)?|analysis|reasoning)\b",
    "thinking_channel": (
        r"\[(?:analysis|reasoning|thinking)\]|<\|(?:analysis|thinking)\|>"
    ),
    "thinking_heading": r"^[ \t]*(?:thinking|reasoning|analysis)[ \t]*:",
    "thinking_tool": r"^[ \t]*(?:[\w.]*think[\w.]*|scratchpad)\(",
    "reasoning_json_key": r'"(?:reasoning|reasoning_content|thinking)"\s*:',
}


def sections(text: str) -> list[tuple[str, int, int]]:
    """Identify rendered message bodies in the two source transcript formats."""
    matches = list(HEADERS.finditer(text))
    return [
        (
            m[1] or m[2],
            m.end(),
            matches[i + 1].start() if i + 1 < len(matches) else len(text),
        )
        for i, m in enumerate(matches)
    ]


def location(text: str, position: int) -> str:
    """Classify an overlap by visible message role and submitted code fences."""
    role = next(
        (role for role, start, end in sections(text) if start <= position < end),
        "unknown",
    )
    if role == "TOOL CALLS":
        start = text.rfind("\n", 0, position) + 1
        if text[start:position].startswith("submit("):
            return "submit_argument"
    if text[:position].count("```") % 2:
        return "code_fence"
    return role.lower()


def paragraph_overlaps(
    thinking: str, transcript: str, minimum: int
) -> list[dict[str, Any]]:
    """Find exact long paragraphs and retain their location for manual review."""
    hits = []
    for paragraph in re.split(r"\n\s*\n", thinking):
        paragraph = paragraph.strip()
        if len(paragraph) >= minimum and paragraph in transcript:
            pos = transcript.index(paragraph)
            hits.append(
                {
                    "chars": len(paragraph),
                    "sha256": digest(paragraph),
                    "location": location(transcript, pos),
                    "text": paragraph,
                }
            )
    return hits


def main() -> None:
    config = json.loads(Path(__file__).with_name("config.json").read_text())
    teacher, student = {}, {}
    for name, output in [("teacher", teacher), ("student", student)]:
        path = Path(config[name + "_input"])
        if sha256_file(path) != config[name + "_sha256"]:
            raise ValueError("Frozen prompt file drift")
        for row in read_jsonl(path):
            if (
                row["id"] in output
                or digest(row["prompt"]) != row["metadata"]["rendered_prompt_sha256"]
            ):
                raise ValueError("Prompt identity drift")
            output[row["id"]] = row
    if len(teacher) != config["rows"] or set(teacher) != set(student):
        raise ValueError("Prompt coverage drift")
    source_rows, provenance = load_source_rows(SOURCE_DIRECTORY)
    originals = {(r["source"], r["source_row_index"]): r for r in source_rows}
    template = load_prompt_set()
    by_source = defaultdict(Counter)
    for key, row in teacher.items():
        meta = row["metadata"]
        original = originals[(meta["source_dataset"], meta["source_row_index"])]
        text = original["trajectory"]
        if (
            template.teacher.render(text) != row["prompt"]
            or template.student.render(text) != student[key]["prompt"]
        ):
            raise ValueError("Source/teacher/student trajectory mismatch")
        if (
            digest(text) != meta["trajectory_sha256"]
            or meta["ground_truth"] != original["ground_truth"]
        ):
            raise ValueError("Source trajectory or label drift")
        by_source[original["source"]]["exact_source_teacher_student_matches"] += 1
    cached = set()
    for path in config["teacher_caches"]:
        for row in read_jsonl(Path(path)):
            if row["id"] in cached or row["prompt_sha256"] != digest(
                teacher[row["id"]]["prompt"]
            ):
                raise ValueError("Actual teacher cache mismatch")
            cached.add(row["id"])
    if cached != set(teacher):
        raise ValueError("Actual teacher cache coverage mismatch")
    details = []
    for source in OOD_SOURCES:
        counts = by_source[source.source]
        for row in pq.read_table(SOURCE_DIRECTORY / source.filename).to_pylist():
            text = row["transcript"]
            counts["rows"] += 1
            record = {
                "source": source.source,
                "row_index": row["row_index"],
                "markers": [],
                "source_content_overlaps": [],
                "thinking_paragraph_overlaps": [],
            }
            for kind, pattern in MARKERS.items():
                for match in re.finditer(pattern, text, re.I | re.M):
                    loc = location(text, match.start())
                    counts["marker_" + kind + "_" + loc] += 1
                    record["markers"].append(
                        {
                            "kind": kind,
                            "location": loc,
                            "context": text[
                                max(0, match.start() - 80) : match.end() + 300
                            ],
                        }
                    )
            for role, start, end in sections(text):
                if role != "ASSISTANT":
                    continue
                body = text[start:end].strip()
                counts["assistant_sections"] += 1
                counts["nonempty_assistant_sections"] += bool(body)
                if source.source == "cua_shade_arena":
                    counts["assistant_tool_names_only"] += bool(
                        re.fullmatch(r"Tool calls:\s*(?:- [\w.-]+\s*)+", body)
                    )
            if "trajectory_data" in row:
                data = json.loads(row["trajectory_data"])
                for message in data["messages"]:
                    if message["role"] != "assistant":
                        continue
                    counts["raw_assistant_messages"] += 1
                    content = message.get("content")
                    if isinstance(content, str) and content.strip():
                        content = content.strip()
                        counts["raw_nonempty_assistant_content"] += 1
                        if content in text:
                            loc = location(text, text.index(content))
                            counts["raw_content_overlap_" + loc] += 1
                            record["source_content_overlaps"].append(
                                {
                                    "chars": len(content),
                                    "location": loc,
                                    "text": content,
                                }
                            )
                    elif isinstance(content, list):
                        raise ValueError("New typed source layout requires review")
                    counts["raw_explicit_think_calls"] += len(
                        re.findall(
                            r"function=['\"]think['\"]", str(message.get("tool_calls"))
                        )
                    )
            if "thinking" in row:
                thinking = row["thinking"] or ""
                counts["separate_thinking_nonempty"] += bool(thinking.strip())
                counts["complete_thinking_present"] += bool(
                    thinking.strip() and thinking.strip() in text
                )
                steps = [
                    s.strip()
                    for s in re.split(r"^--- Step .*?---\s*$", thinking, flags=re.M)
                    if s.strip()
                ]
                counts["thinking_steps"] += len(steps)
                counts["complete_thinking_steps_present"] += sum(
                    s in text for s in steps
                )
                counts["long_thinking_paragraphs"] += sum(
                    len(p.strip()) >= config["minimum_overlap_paragraph_chars"]
                    for p in re.split(r"\n\s*\n", thinking)
                )
                hits = paragraph_overlaps(
                    thinking, text, config["minimum_overlap_paragraph_chars"]
                )
                record["thinking_paragraph_overlaps"] = hits
                counts["rows_with_thinking_paragraph_overlap"] += bool(hits)
                for hit in hits:
                    counts["thinking_paragraph_overlap_" + hit["location"]] += 1
            details.append(record)
    output = Path("results/ood_reasoning_audit")
    result = {
        "config": config,
        "source_provenance": provenance,
        "teacher_cache_exact_matches": len(cached),
        "counts": dict(by_source),
    }
    atomic_write_json(output / "summary.json", result)
    atomic_write_jsonl(output / "rows.jsonl", details)
    print(json.dumps(result["counts"], indent=2))


if __name__ == "__main__":
    main()
