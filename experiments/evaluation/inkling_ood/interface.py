"""Inkling 2 Mini's direct-decision prompt boundary. No heavy imports: the CPU
audit, the tests and the Pod scorer all share this module."""

from __future__ import annotations

from typing import Any

MODEL_ID = "thinkingmachines/Inkling-2-Mini"
REVISION = "7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6"
USER_END = "<|end_message|>"
# The template has no generation-prompt branch: it stops after the last user
# turn. An assistant text answer opens with these two markers.
ASSISTANT_OPEN = "<|message_model|><|content_text|>"
MODEL_MARKER = "<|message_model|>"
DECISION_PREFIX = "Prediction:"
SPECIAL_MARKERS = (
    "<|endoftext|>",
    "<|message_user|>",
    "<|message_model|>",
    "<|message_system|>",
    "<|message_tool|>",
    "<|content_text|>",
    "<|content_image|>",
    "<|content_model_end_sampling|>",
    "<|content_thinking|>",
    "<|end_message|>",
    "<|content_tool_error|>",
    "<|content_xml|>",
    "<|begin_of_text|>",
    "<|content_invoke_tool_json|>",
    "<|content_invoke_tool_text|>",
)


def render_chat(tokenizer: Any, user_prompt: str, reasoning_effort: str) -> str:
    """The template's own rendering of one user turn at ``reasoning_effort``.

    The template writes the effort as a system line before the first
    non-system message and appends nothing after the user turn.
    """
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort=reasoning_effort,
    )
    if not rendered.endswith(USER_END) or not rendered.startswith(
        "<|message_system|><|content_text|>Thinking effort level: "
    ):
        raise ValueError("Inkling chat template no longer ends at the user turn")
    return rendered


def render_margin_prompt(
    tokenizer: Any,
    user_prompt: str,
    *,
    reasoning_effort: str,
    assistant_open: str = ASSISTANT_OPEN,
    decision_prefix: str = DECISION_PREFIX,
) -> str:
    """Render one user turn and open a text answer at the decision prefix."""
    rendered = render_chat(tokenizer, user_prompt, reasoning_effort)
    return f"{rendered}{assistant_open}{decision_prefix}"


def opening_prompt(tokenizer: Any, user_prompt: str, *, reasoning_effort: str) -> str:
    """Prompt ending at the bare model marker, for the unconstrained diagnostic:
    does Inkling open ``<|content_thinking|>`` or ``<|content_text|>``?"""
    return render_chat(tokenizer, user_prompt, reasoning_effort) + MODEL_MARKER


def contains_special_marker(user_prompt: str) -> bool:
    """True when user text contains a literal Inkling control string. The
    template does not escape it, so the tokenizer would read it as a marker."""
    return any(marker in user_prompt for marker in SPECIAL_MARKERS)
