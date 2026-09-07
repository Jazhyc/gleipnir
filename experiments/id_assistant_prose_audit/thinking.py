"""Use structured STRIDE provenance to remove thinking calls losslessly."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from experiments.id_assistant_prose_audit.run import HEADER, strip_assistant_prose


def strip_thinking_calls(
    trajectory: str,
    tool_blocks: list[list[tuple[str, str]]],
    *,
    remove_prose: bool = True,
) -> tuple[str, int]:
    """Validate exact source call blocks, then remove only named think calls."""
    matches = list(HEADER.finditer(trajectory))
    output, removed, block_index = [], 0, 0
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(trajectory)
        chunk = trajectory[match.start() : end]
        if match[1].lower() == "tool calls":
            body = trajectory[match.end() : end]
            calls = tool_blocks[block_index]
            block_index += 1
            if body.rstrip() != "\n".join(text for _, text in calls):
                raise ValueError("Rendered calls differ from structured source")
            kept = [text for name, text in calls if name != "think"]
            removed += sum(name == "think" for name, _ in calls)
            if not kept:
                if i + 1 < len(matches) and matches[i + 1][1].lower() == "tool":
                    raise ValueError(
                        "Thinking-only call has an unattributed tool result"
                    )
                chunk = ""
            else:
                chunk = match[0] + "\n".join(kept) + body[len(body.rstrip()) :]
        output.append(chunk)
    if block_index != len(tool_blocks):
        raise ValueError("Structured source call count drift")
    result = "".join(output)
    return (strip_assistant_prose(result)[0] if remove_prose else result), removed


def source_records() -> dict[str, dict[str, Any]]:
    """Read typed reasoning and tool calls from the raw STRIDE source."""
    path = Path(
        "data/tool_trajectory_monitoring/source/id_evaluation/stride_test.parquet"
    )
    expected_sha256 = "c13af47d00d3a32e9fcecb13df01e9f8bdf6a80e956d25ed00e8cb15dcc1d43b"
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("Structured STRIDE source checksum drift")
    output = {}
    for row in pq.read_table(path, columns=["id", "trajectory_data"]).to_pylist():
        texts, tool_blocks, assistant_blocks = [], [], []
        for message in json.loads(row["trajectory_data"])["messages"]:
            if message["role"] != "assistant":
                continue
            blocks = [
                (
                    block.get("type"),
                    block.get("reasoning", block.get("text", "")).strip(),
                )
                for block in message.get("content") or []
                if isinstance(block, dict)
            ]
            blocks = [(kind, text) for kind, text in blocks if text]
            if blocks or message.get("tool_calls"):
                assistant_blocks.append(blocks)
            for block in message.get("content") or []:
                if isinstance(block, dict) and block.get("type") == "reasoning":
                    text = block.get("reasoning", "").strip()
                    if text:
                        texts.append(text)
            if message.get("tool_calls"):
                calls = []
                for call in message["tool_calls"]:
                    args = call["arguments"]
                    rendered = (
                        json.dumps(args, ensure_ascii=False)
                        if isinstance(args, dict)
                        else str(args).rstrip()
                    )
                    calls.append((call["function"], f"{call['function']}({rendered})"))
                tool_blocks.append(calls)
        output["test_stride:" + row["id"]] = {
            "reasoning": texts,
            "assistant_blocks": assistant_blocks,
            "tool_blocks": tool_blocks,
        }
    return output
