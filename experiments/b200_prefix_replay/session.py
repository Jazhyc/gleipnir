"""Synthetic long monitored session from consecutive Gloom trajectories."""

from __future__ import annotations

import json
import random
from pathlib import Path

SENTINEL = "<<GLEIPNIR_CONTENT>>"
TAIL = "\n</agent_trajectory>\n"
ESCALATION_RATE = 0.072  # OpenAI Auto-review snapshot: 720/10,000 actions reviewed.


def split_steps(trajectory: str) -> tuple[str, list[str]]:
    """Header up to the first assistant turn, then one block per assistant turn."""
    marker = "\n[ASSISTANT]\n"
    first = trajectory.find(marker)
    if first < 0:
        raise ValueError("trajectory has no assistant turn")
    header, rest = trajectory[: first + 1], trajectory[first + 1 :]
    parts = rest.split(marker)
    blocks = [parts[0]] + [marker[1:] + p for p in parts[1:]]
    return header, blocks


def action_part(block: str) -> str | None:
    """The proposed tool call: the block up to its first tool output."""
    if "[TOOL CALLS]" not in block:
        return None
    end = block.find("\n[TOOL]\n")
    return block if end < 0 else block[: end + 1]


def build(rows: list[dict], count, limit: int) -> dict:
    """Chain step blocks until the rendered prompt would exceed ``limit`` tokens."""
    header = None
    steps, total = [], 0
    for row in rows:
        prompt = row["student_prompt"]
        body = prompt[: -len(TAIL) + 1] if prompt.endswith(TAIL[1:]) else prompt
        head, blocks = split_steps(body)
        if header is None:
            header = head
            total = count(header) + count(TAIL)
        for block in blocks:
            action = action_part(block)
            tokens = count(block)
            if total + tokens > limit:
                return {"header": header, "steps": steps, "tokens_estimate": total}
            steps.append(
                {"block": block, "action": action, "source_index": row["index"]}
            )
            total += tokens
    raise ValueError("not enough Gloom steps to reach the limit")


def policies(steps: list[dict], seed: int = 0) -> dict[str, list[int]]:
    actions = [i for i, s in enumerate(steps) if s["action"] is not None]
    rng = random.Random(seed)
    escalated = [i for i in actions if rng.random() < ESCALATION_RATE]
    return {"every_action": actions, "escalated": escalated}


def content(session: dict, index: int) -> str:
    """Monitor content for a check at step ``index``: history plus its action."""
    steps = session["steps"]
    history = "".join(s["block"] for s in steps[:index])
    return session["header"] + history + steps[index]["action"] + TAIL


def render(session: dict, index: int) -> str:
    return session["template"].replace(SENTINEL, content(session, index))


def load(path: Path) -> dict:
    return json.loads(path.read_text())
