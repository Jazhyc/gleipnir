"""Preserve source evidence during original ControlArena export/comparison."""

import json
import zipfile

from experiments.ood_visible_prose.compare_original_control import (
    canonical,
    export_archive,
    identity,
)


def test_normalization_preserves_nonempty_reasoning_and_inline_literals():
    messages = [
        {
            "role": "assistant",
            "content": "<think></think>\nPublic report",
            "tool_calls": [],
        },
        {
            "role": "assistant",
            "content": "<think>private</think>Report",
            "tool_calls": [],
        },
        {
            "role": "assistant",
            "content": "Quoted <think></think> literal",
            "tool_calls": [],
        },
        {"role": "assistant", "content": "", "tool_calls": []},
    ]
    result = canonical(messages, True, True)
    assert len(result) == 3
    assert result[0][1] == "Public report"
    assert result[1][1] == "<think>private</think>Report"
    assert result[2][1] == "Quoted <think></think> literal"


def test_changed_tool_payload_does_not_match():
    a = [
        {
            "role": "assistant",
            "content": "report",
            "tool_calls": [{"name": "submit", "arguments": {"answer": "done"}}],
        }
    ]
    b = [
        {
            "role": "assistant",
            "content": "report",
            "tool_calls": [
                {"id": "x", "function": "submit", "arguments": {"answer": "changed"}}
            ],
        }
    ]
    assert identity(canonical(a, True, True)) != identity(canonical(b, False, True))


def test_native_archive_export_retains_typed_content(tmp_path):
    path = tmp_path / "original.eval"
    output = tmp_path / "export.jsonl"
    blocks = [
        {"type": "reasoning", "reasoning": "private", "signature": "original"},
        {"type": "text", "text": "public"},
    ]
    sample = {
        "id": "sample",
        "epoch": 1,
        "messages": [{"role": "assistant", "content": blocks}],
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "header.json", json.dumps({"eval": {"dataset": {"name": "test"}}})
        )
        archive.writestr("samples/sample_epoch_1.json", json.dumps(sample))
        archive.writestr("_journal/unrelated.json", "{}")
    receipt = export_archive(path, output)
    assert json.loads(output.read_text())["messages"] == sample["messages"]
    assert receipt["counts"]["samples"] == 1
    assert receipt["counts"]["block:reasoning"] == 1
    assert receipt["counts"]["block:text"] == 1
