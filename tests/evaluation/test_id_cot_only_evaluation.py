"""CoT removal preserves visible duplicates and action evidence."""

import pytest

from experiments.id_cot_only_evaluation.prepare import strip_cot


def test_remove_reasoning_copy_keep_visible_copy_and_action():
    trajectory = (
        "[USER]\nTask\n\n[ASSISTANT]\nsame\nsame\n\n[TOOL CALLS]\n"
        'think({"thought": "private"})\n'
        'run({"command": "echo same"})\n\n[TOOL]\nsame\n'
    )
    raw = {
        "assistant_blocks": [[("reasoning", "same"), ("text", "same")]],
        "tool_blocks": [
            [
                ("think", 'think({"thought": "private"})'),
                ("run", 'run({"command": "echo same"})'),
            ]
        ],
    }
    result, calls, blocks = strip_cot(trajectory, raw)
    assert (
        result == "[USER]\nTask\n\n[ASSISTANT]\n\nsame\n\n[TOOL CALLS]\n"
        'run({"command": "echo same"})\n\n[TOOL]\nsame\n'
    )
    assert (calls, blocks) == (1, 1)


def test_visible_prose_is_unchanged():
    trajectory = "[ASSISTANT]\nI will do the task.\n"
    raw = {"assistant_blocks": [[("text", "I will do the task.")]], "tool_blocks": []}
    assert strip_cot(trajectory, raw) == (trajectory, 0, 0)


def test_fail_closed_on_source_mismatch():
    raw = {"assistant_blocks": [[("reasoning", "different")]], "tool_blocks": []}
    with pytest.raises(ValueError, match="differs"):
        strip_cot("[ASSISTANT]\nprivate\n", raw)
