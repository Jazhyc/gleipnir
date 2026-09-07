"""Check prose removal without changing action or context evidence."""

import pytest

from experiments.id_assistant_prose_audit.run import strip_assistant_prose


@pytest.mark.parametrize(
    "header,call,result",
    [
        ("ASSISTANT", "TOOL CALLS", "TOOL"),
        ("assistant", "tool_call: think", "tool_result for call_1"),
    ],
)
def test_preserves_actions_and_results(header, call, result):
    prefix = "[USER]\nKeep user prose.\n\n"
    action = f'[{call}]\nthink({{"thought": "Keep argument prose"}})\n\n'
    response = f"[{result}]\nKeep tool prose.\n\n"
    trajectory = (
        prefix
        + f"[{header}]\nRemove mixed prose.\n\n"
        + action
        + response
        + f"[{header}]\nRemove standalone prose.\n"
    )
    stripped, count = strip_assistant_prose(trajectory)
    assert stripped == prefix + f"[{header}]\n" + action + response
    assert count == 2
    assert strip_assistant_prose(stripped) == (stripped, 0)


def test_keeps_context_after_prose_only_turn():
    trajectory = "[USER]\nTask\n\n[ASSISTANT]\nClarify\n\n[USER]\nAnswer\n"
    assert strip_assistant_prose(trajectory) == ("[USER]\nTask\n\n[USER]\nAnswer\n", 1)


def test_unknown_format_fails():
    with pytest.raises(ValueError):
        strip_assistant_prose("ASSISTANT: unknown")


def test_multiline_think_and_mixed_real_call():
    from experiments.id_assistant_prose_audit.thinking import strip_thinking_calls

    context = "[USER]\nTask\n\n"
    trajectory = (
        context + "[ASSISTANT]\n[TOOL CALLS]\n"
        'think({"thought": "First\nthen second"})\n'
        'bash({"cmd": "echo hello"})\n\n[TOOL]\nhello\n'
    )
    assert strip_thinking_calls(
        trajectory,
        [
            [
                ("think", 'think({"thought": "First\nthen second"})'),
                ("bash", 'bash({"cmd": "echo hello"})'),
            ]
        ],
    ) == (
        context + "[ASSISTANT]\n[TOOL CALLS]\n"
        'bash({"cmd": "echo hello"})\n\n[TOOL]\nhello\n',
        1,
    )


def test_thinking_only_turn_removed():
    from experiments.id_assistant_prose_audit.thinking import strip_thinking_calls

    trajectory = "[USER]\nTask\n\n[ASSISTANT]\n[TOOL CALLS]\nthink({})\n"
    assert strip_thinking_calls(trajectory, [[("think", "think({})")]]) == (
        "[USER]\nTask\n\n",
        1,
    )


def test_source_validation_handles_unescaped_source_text():
    from experiments.id_assistant_prose_audit.thinking import strip_thinking_calls

    # Native source strings are not necessarily valid JSON or Python literals.
    thought = "think({'thought': 'I'll inspect it.'})"
    action = 'python({"code": "print(1)\nprint(2)"})'
    trajectory = (
        "[USER]\nTask\n\n[ASSISTANT]\n[TOOL CALLS]\n" + thought + "\n" + action + "\n"
    )
    stripped, removed = strip_thinking_calls(
        trajectory, [[("think", thought), ("python", action)]]
    )
    assert stripped == "[USER]\nTask\n\n[ASSISTANT]\n[TOOL CALLS]\n" + action + "\n"
    assert removed == 1
    with pytest.raises(ValueError, match="structured source"):
        strip_thinking_calls(trajectory, [[("think", "think({})")]])
