"""The released student prompt contract (tool-trajectory-binary-v1), torch-free.

A row's user prompt is `instruction + OPEN + trajectory + CLOSE`; the student
instruction ends in a newline, so a blank line precedes OPEN. The monitor sees
it as one user turn under the Qwen chat template with thinking disabled,
followed by `Prediction:`. Attacks edit only the trajectory and rejoin it with
the row's own instruction, byte for byte.
"""

from __future__ import annotations

from typing import Any, NamedTuple

OPEN = "\n<agent_trajectory>\n"
CLOSE = "\n</agent_trajectory>\n"
DECISION_PREFIX = "Prediction:"
QWEN_NON_THINKING_ASSISTANT_SUFFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
MAX_TOKENS = 29_696  # training max_length of all three released monitors


class Parts(NamedTuple):
    head: str
    trajectory: str


def split_prompt(prompt: str) -> Parts:
    """Split at the first OPEN and the last CLOSE, so tags inside survive."""
    head, sep, rest = prompt.partition(OPEN)
    if not sep or not rest.endswith(CLOSE):
        raise ValueError("prompt does not follow the trajectory contract")
    return Parts(head, rest[: -len(CLOSE)])


def join_prompt(head: str, trajectory: str) -> str:
    return f"{head}{OPEN}{trajectory}{CLOSE}"


def render(tokenizer: Any, user_prompt: str) -> str:
    """Match the model cards and `render_margin_prompt` in the OOD benchmark."""
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not rendered.endswith(QWEN_NON_THINKING_ASSISTANT_SUFFIX):
        raise ValueError("chat template no longer ends at the non-thinking boundary")
    return rendered + DECISION_PREFIX
