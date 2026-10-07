"""Distinguish source thinking from visible tool evidence."""

import re

from experiments.ood_reasoning_audit.run import MARKERS, location, paragraph_overlaps


def test_heading_in_tool_output_is_not_an_assistant_trace():
    text = "USER: Task\nASSISTANT: Tool calls:\n- run\nTOOL: Analysis: result\n"
    assert location(text, text.index("Analysis:")) == "tool"
    assert location("[ASSISTANT]\n[TOOL CALLS]\nthink({})", 25) == "tool calls"


def test_submitted_answer_and_code_overlap_are_distinguished():
    text = '[ASSISTANT]\n[TOOL CALLS]\nsubmit({"answer": "Finished"})\n'
    assert location(text, text.index("Finished")) == "submit_argument"
    code = 'print("visible code")'
    hits = paragraph_overlaps(code, "[ASSISTANT]\nCode:\n```\n" + code + "\n```", 10)
    assert hits[0]["location"] == "code_fence"


def test_explicit_thinking_marker_and_minimum_overlap():
    assert re.search(MARKERS["thinking_tool"], "sequentialthinking({})", re.M)
    assert re.search(MARKERS["thinking_tag"], "<think>private</think>")
    assert not paragraph_overlaps("I", "[USER]\nI need help", 80)
