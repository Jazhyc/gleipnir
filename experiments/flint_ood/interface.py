"""Flint's direct-decision prompt boundary. No heavy imports: the CPU audit,
the tests and the Pod scorer all share this module."""

from __future__ import annotations

from typing import Any

MODEL_ID = "RecursiveSI/flint"
REVISION = "2a247be0c0aec895ccb0ac7ca92229ec6df37288"
ASSISTANT_SUFFIX = "<|assistant|>"
DECISION_PREFIX = "Prediction:"
# The template writes reasoning as <think>…</think> directly before the answer.
EMPTY_THINK = "<think></think>"


def render_margin_prompt(
    tokenizer: Any,
    user_prompt: str,
    *,
    assistant_suffix: str = ASSISTANT_SUFFIX,
    assistant_prefill: str = "",
    decision_prefix: str = DECISION_PREFIX,
) -> str:
    """Render one user turn and open the assistant turn at the decision prefix.

    Flint has no thinking switch: it chooses per turn whether to open with
    ``<think>``. Starting the assistant turn at ``Prediction:`` scores the turn
    in which it answers directly. ``assistant_prefill`` goes between the
    assistant marker and the prefix; ``EMPTY_THINK`` there closes an empty
    reasoning block in the template's own format. The template uses no BOS.
    """
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if not rendered.endswith(assistant_suffix) or rendered.count(assistant_suffix) != 1:
        raise ValueError("Flint chat template no longer ends at one assistant marker")
    return f"{rendered}{assistant_prefill}{decision_prefix}"


def template_changed_user_text(rendered: str, user_prompt: str) -> bool:
    """True when the template's sanitizer rewrote the user text (it removes NUL
    and think tags and escapes role markers)."""
    return user_prompt not in rendered
