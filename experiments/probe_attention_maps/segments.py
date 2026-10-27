"""Label trajectory characters and tokens by conversational segment.

OOD trajectories use two header styles: bracketed lines (``[USER]``,
``[SYSTEM]``, ``[ASSISTANT]``, ``[TOOL CALLS]``, ``[TOOL]``; AgentDojo, IaC,
SHADE-Arena, rogue_eval, MLE sabotage) and line-start colon prefixes
(``USER:``, ``ASSISTANT:``, ``TOOL:``; CUA SHADE-Arena, where an assistant
turn that opens with ``Tool calls:`` is a tool call). Header text itself is
its own ``marker`` class; text before the first header is ``other``.
"""

from __future__ import annotations

import re
from bisect import bisect_right

CLASSES = ("other", "marker", "system", "user", "assistant", "tool_call", "tool_output")
BRACKET = re.compile(r"^\[(SYSTEM|USER|ASSISTANT|TOOL CALLS|TOOL)\][ \t]*$", re.M)
COLON = re.compile(r"^(SYSTEM|USER|ASSISTANT|TOOL):[ \t]?", re.M)
ROLE = {
    "SYSTEM": "system",
    "USER": "user",
    "ASSISTANT": "assistant",
    "TOOL CALLS": "tool_call",
    "TOOL": "tool_output",
}


def spans(text: str) -> list[tuple[int, str]]:
    """Sorted ``(start_char, class)`` boundaries covering ``text`` from 0."""
    out = [(0, "other")]
    if BRACKET.search(text):
        for m in BRACKET.finditer(text):
            out += [(m.start(), "marker"), (m.end(), ROLE[m.group(1)])]
        return out
    for m in COLON.finditer(text):
        role = ROLE[m.group(1)]
        if role == "assistant" and text.startswith("Tool calls:", m.end()):
            role = "tool_call"
        out += [(m.start(), "marker"), (m.end(), role)]
    return out


def token_classes(text: str, offsets: list[tuple[int, int]]) -> list[int]:
    """Class index (into ``CLASSES``) of each token, by its first character."""
    bounds = spans(text)
    starts = [s for s, _ in bounds]
    index = {c: i for i, c in enumerate(CLASSES)}
    return [index[bounds[bisect_right(starts, start) - 1][1]] for start, _ in offsets]
