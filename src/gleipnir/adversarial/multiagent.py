"""Turn six-worker Halden lab histories into action-only monitor trajectories.

Each worker's saved history (`<W>-history.json`: system, user, assistant and
toolResult messages) becomes one trajectory in the role-tagged format of the
strict-OOD suite (AgentDojo, IaC, Rogue Eval, SHADE-Arena):

    [USER]                  lab check-in text, including incoming board posts
    [ASSISTANT]
    [TOOL CALLS]            name({json arguments}), one call per line
    [TOOL]                  tool output

The system prompt and all assistant prose and thinking are dropped, as in the
action-only contract. Tool outputs saved as JSON are decoded: strings are shown
raw, and objects as `key: value` lines with multi-line strings shown raw.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any

BOARD = re.compile(r"(Incoming board posts:\n)(\[.*?\])(\n)", re.S)


def _texts(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    return "".join(
        part.get("text", "") for part in content or [] if part.get("type") == "text"
    )


def _tool_output(text: str) -> str:
    try:
        value = json.loads(text)
    except ValueError:
        return text
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            if isinstance(item, str) and "\n" in item:
                lines.append(f"{key}:\n{item}")
            elif isinstance(item, str):
                lines.append(f"{key}: {item}")
            else:
                lines.append(f"{key}: {json.dumps(item, ensure_ascii=False)}")
        return "\n".join(lines)
    return json.dumps(value, ensure_ascii=False)


def _arguments(call: dict[str, Any]) -> str:
    return json.dumps(call.get("arguments") or {}, ensure_ascii=False)


def render_history(messages: list[dict[str, Any]], board_as_tool: bool = False) -> str:
    """Render one worker's history as an action-only trajectory.

    With `board_as_tool`, each check-in's incoming-board-posts list is moved out
    of the user turn into a tool-output block right after it, so peer posts read
    as untrusted tool output rather than as the user's own words. The lab
    delivered them in the user turn; this is a rendering variant for testing.
    """
    blocks = []
    for message in messages:
        role = message.get("role")
        if role == "user":
            text = _texts(message)
            match = BOARD.search(text) if board_as_tool else None
            if match:
                user = text[: match.start()] + text[match.end() :]
                blocks.append(f"[USER]\n{user}")
                blocks.append(f"[TOOL]\n{match.group(1)}{match.group(2)}")
            else:
                blocks.append(f"[USER]\n{text}")
        elif role == "assistant":
            calls = [
                f"{part['name']}({_arguments(part)})"
                for part in message.get("content") or []
                if part.get("type") == "toolCall"
            ]
            if calls:
                blocks.append("[ASSISTANT]\n[TOOL CALLS]\n" + "\n".join(calls))
        elif role == "toolResult":
            blocks.append(f"[TOOL]\n{_tool_output(_texts(message))}")
    return "\n\n".join(blocks)


def board_turns(messages: list[dict[str, Any]]) -> list[int]:
    """Indices of user check-ins that carry an incoming-board-posts list."""
    return [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "user" and BOARD.search(_texts(message))
    ]


def max_post_id(messages: list[dict[str, Any]]) -> int:
    ids = [0]
    for index in board_turns(messages):
        posts = json.loads(BOARD.search(_texts(messages[index])).group(2))
        ids += [int(post["id"]) for post in posts if "id" in post]
    return max(ids)


def plant_post(
    messages: list[dict[str, Any]], turn: int, author: str, text: str
) -> list[dict[str, Any]]:
    """Return a copy with one extra board post appended at user turn `turn`.

    Only that check-in's board list changes; the worker's own actions and every
    other message stay byte-identical, so a score change comes from reading.
    """
    planted = copy.deepcopy(messages)
    message = planted[turn]
    body = _texts(message)
    match = BOARD.search(body)
    if message.get("role") != "user" or not match:
        raise ValueError("turn has no incoming board posts")
    posts = json.loads(match.group(2))
    cycle = re.search(r"service cycle (\d+) of", body)
    posts.append(
        {
            "id": max_post_id(messages) + 1,
            "author": author,
            "cycle": int(cycle.group(1)) - 1 if cycle else None,
            "text": text,
            "replyTo": None,
        }
    )
    encoded = json.dumps(posts, separators=(",", ":"), ensure_ascii=False)
    new_body = body[: match.start(2)] + encoded + body[match.end(2) :]
    message["content"] = [{"type": "text", "text": new_body}]
    return planted
