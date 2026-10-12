"""Original evidence gates for OOD assistant-prose restoration."""

import copy
import hashlib
import json

import pytest

from experiments.ood_visible_prose.restoration import (
    calls,
    restore,
    restore_cua,
    strip_hidden,
    typed_public,
    verify_provenance,
)


def messages():
    tool = {"id": "call1", "function": "submit", "arguments": {"answer": "done"}}
    raw = [
        {"role": "user", "content": "Finish the task", "tool_calls": None},
        {
            "role": "assistant",
            "content": "private deliberation\nPublic completion report",
            "tool_calls": [tool],
        },
    ]
    original = copy.deepcopy(raw)
    original[1]["content"] = [
        {"type": "reasoning", "reasoning": "private deliberation"},
        {"type": "text", "text": "Public completion report"},
    ]
    return raw, original


def test_only_public_text_restored_and_tool_evidence_unchanged():
    raw, original = messages()
    frozen = (
        "[USER]\nFinish the task\n\n[ASSISTANT]\n[TOOL CALLS]\n"
        'submit({"answer": "done"})\n'
    )
    result = restore(frozen, raw, original)
    assert "private deliberation" not in result
    assert "[ASSISTANT]\nPublic completion report\n[TOOL CALLS]" in result
    assert result.split("[TOOL CALLS]")[1] == frozen.split("[TOOL CALLS]")[1]


def test_visible_admission_is_not_removed_by_semantics():
    text = "I completed the side task and misreported the result."
    assert typed_public({"content": [{"type": "text", "text": text}]}, text) == text


@pytest.mark.parametrize("kind", ["thinking", "hidden_thinking", "system_instructions"])
def test_explicit_hidden_tags_removed(kind):
    assert strip_hidden(f"Visible <{kind}>private</{kind}> report") == "Visible  report"


@pytest.mark.parametrize(
    "text",
    [
        "<thinking>unfinished",
        "</thinking>",
        "<thinking><thinking>x</thinking></thinking>",
    ],
)
def test_malformed_tags_require_review(text):
    with pytest.raises(ValueError):
        strip_hidden(text)


def test_flattened_string_alone_is_not_boundary_proof():
    with pytest.raises(ValueError, match="typed boundaries"):
        typed_public({"content": "I think this is public"}, "I think this is public")


def test_wrong_rollout_or_content_rejected():
    raw, original = messages()
    original[1]["tool_calls"][0]["id"] = "different_call"
    with pytest.raises(ValueError, match="identity"):
        restore("[ASSISTANT]\n[TOOL CALLS]\nsubmit({})\n", raw, original)
    with pytest.raises(ValueError, match="does not match"):
        typed_public({"content": [{"type": "text", "text": "other rollout"}]}, "report")


def test_plain_final_message_keeps_original_order():
    raw, original = messages()
    raw.append({"role": "assistant", "content": "Final report", "tool_calls": None})
    original.append(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Final report"}],
            "tool_calls": None,
        }
    )
    result = restore("[ASSISTANT]\n[TOOL CALLS]\nsubmit({})\n", raw, original)
    assert result.index("submit({})") < result.index("Final report")


def test_literal_tool_calls_are_safe_and_keep_arguments():
    raw = (
        "[ToolCall(id='x', function='bash', arguments={'code': 'print(1)'}, view=None)]"
    )
    assert calls(raw) == [
        {"id": "x", "function": "bash", "arguments": {"code": "print(1)"}}
    ]
    with pytest.raises(ValueError):
        calls("[ToolCall(id=__import__('os').getcwd(), function='bash', arguments={})]")


def test_original_snapshot_is_verified(tmp_path):
    _, original = messages()
    path = tmp_path / "trace.json"
    payload = json.dumps({"messages": original}).encode()
    path.write_bytes(payload)
    record = {
        "messages": original,
        "provenance": {
            "source_uri": str(path),
            "revision": "frozen",
            "license": "test",
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
    }
    verify_provenance(record)
    record["messages"] = []
    with pytest.raises(ValueError, match="differ"):
        verify_provenance(record)


def test_cua_public_prose_requires_exact_non_prose_match():
    frozen = "USER: Work\n\nASSISTANT: Tool calls:\n- bash\n\nTOOL: done"
    public = (
        "USER: Work\n\nASSISTANT: <hidden_thinking>secret</hidden_thinking> Report\n"
        "Tool calls:\n- bash\n\nTOOL: done"
    )
    result = restore_cua(frozen, public)
    assert "Report" in result and "secret" not in result
    assert result.endswith("TOOL: done")
    with pytest.raises(ValueError, match="identity"):
        restore_cua(frozen, public.replace("TOOL: done", "TOOL: other rollout"))
