"""Keep typed reasoning distinct from prose and tool-output echoes."""

from experiments.id_assistant_prose_audit.training_audit import (
    count_think_calls,
    typed_reasoning,
)


def test_only_nonempty_assistant_reasoning_is_counted():
    blocks = [
        {"type": "reasoning", "reasoning": "  plan privately  "},
        {"type": "reasoning", "reasoning": " "},
        {"type": "text", "text": "visible report"},
    ]
    assert typed_reasoning(
        [
            {"role": "user", "content": blocks},
            {"role": "assistant", "content": "plain text"},
            {"role": "assistant", "content": blocks},
        ]
    ) == ["plan privately"]


def test_count_ignores_tool_output_echo_and_handles_multiline_thought():
    trajectory = (
        "[USER]\nTask\n\n[ASSISTANT]\n[TOOL CALLS]\n"
        'think({"thought": "first\nsecond"})\n'
        'bash({"cmd": "echo hello"})\n\n[TOOL]\nthink(not a call)\n'
    )
    assert count_think_calls(trajectory) == 1
