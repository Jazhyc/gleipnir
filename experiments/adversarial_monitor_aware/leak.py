"""Check that no added system-prompt text reaches the monitor's rendering (stdlib only).

The monitor reads `adversarial_natural_cheats.trajectories.render`: [USER]
turns, tool calls and tool outputs, never the system prompt or assistant prose.
`check` looks for each added paragraph, each listed distinctive phrase and every
6-word run of the paragraphs (case-insensitive, whitespace-normalized).

- Hits in [USER] blocks come from the scaffold, so any hit is a rendering leak.
- Hits in tool calls or tool outputs can only come from the agent repeating
  what it read; they are behaviour (an "echo"), counted separately.
"""

from __future__ import annotations

import re
from typing import Any

from experiments.adversarial_natural_cheats.trajectories import render

SHINGLE = 6


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def needles(config: dict[str, Any]) -> list[str]:
    """Every string whose presence in a rendering counts as a hit."""
    out: set[str] = set()
    for paragraph in config["paragraphs"].values():
        words = norm(paragraph).split(" ")
        out.add(" ".join(words))
        out.update(
            " ".join(words[i : i + SHINGLE]) for i in range(len(words) - SHINGLE + 1)
        )
    out.update(norm(p) for p in config["distinctive_phrases"])
    return sorted(out)


def hits(text: str, config: dict[str, Any]) -> list[str]:
    haystack = norm(text)
    return [n for n in needles(config) if n in haystack]


def check(messages: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Leak and echo hits in one run's rendered monitor trajectory."""
    rendered = render(messages)
    # Rendering is block-level: check each [USER] message directly, so a
    # blank line inside tool output can't move text between the two sides.
    user_text = "\n".join(m["text"] for m in messages if m["role"] == "user")
    user_hits = hits(user_text, config)
    all_hits = hits(rendered, config)
    return {
        "rendering_leak": user_hits,
        "agent_echo": [h for h in all_hits if h not in user_hits],
    }
