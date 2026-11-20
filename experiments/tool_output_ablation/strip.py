"""Remove tool outputs from a rendered trajectory, keeping its turn structure.

The monitor still sees every user message, assistant message and tool call;
each tool output is replaced by ``OMITTED``. Three renderings occur in the OOD
suite:

- ``[USER]`` / ``[ASSISTANT]`` / ``[TOOL]`` blocks (AgentDojo, IaC, Rogue
  Eval, SHADE-Arena): the body of every ``[TOOL]`` block is replaced.
- ``USER:`` / ``ASSISTANT:`` / ``TOOL:`` turns (CUA-SHADE-Arena): the text
  of every ``TOOL:`` turn is replaced.
- MLE-Sabotage: one ``[ASSISTANT]`` block per step whose ``Execution info:``
  section holds the program output; that section is replaced.
"""

from __future__ import annotations

import re

OMITTED = "[output omitted]"
OPEN, CLOSE = "<agent_trajectory>", "</agent_trajectory>"

_NEXT_BLOCK = r"(?=\n+\[(?:USER|ASSISTANT|SYSTEM|TOOL)\]\n|\n*\Z)"
_NEXT_TURN = r"(?=\n\n(?:USER|ASSISTANT|TOOL|SYSTEM): |\n*\Z)"
_BLOCK = re.compile(r"(\[TOOL\]\n)(.*?)" + _NEXT_BLOCK, re.S)
_TURN = re.compile(r"(?m)(^TOOL: )(.*?)" + _NEXT_TURN, re.S)
_EXECUTION = re.compile(r"(Execution info:\n)(.*?)" + _NEXT_BLOCK, re.S)


def strip_outputs(trajectory: str) -> str:
    """Return ``trajectory`` with every tool output replaced by ``OMITTED``."""
    if "[ASSISTANT]\n" not in trajectory:
        return _TURN.sub(lambda m: m.group(1) + OMITTED, trajectory)
    trajectory = _BLOCK.sub(lambda m: m.group(1) + OMITTED, trajectory)
    return _EXECUTION.sub(lambda m: m.group(1) + OMITTED, trajectory)


def strip_prompt(prompt: str) -> str:
    """Apply ``strip_outputs`` to the trajectory envelope of a full prompt."""
    if prompt.count(OPEN) != 1 or prompt.count(CLOSE) != 1:
        raise ValueError("prompt must contain one trajectory envelope")
    start = prompt.index(OPEN) + len(OPEN)
    end = prompt.index(CLOSE)
    return prompt[:start] + strip_outputs(prompt[start:end]) + prompt[end:]
