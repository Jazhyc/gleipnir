import pytest

from experiments.evaluation.tool_output_ablation.strip import (
    OMITTED,
    strip_outputs,
    strip_prompt,
)


def test_bracket_blocks_keep_calls_and_drop_outputs():
    t = (
        "\n[USER]\npay the bill\n\n[ASSISTANT]\n[TOOL CALLS]\nread_file({})"
        "\n\n[TOOL]\nIBAN: X\n\n[detail]\nmore"
        "\n\n[ASSISTANT]\n[TOOL CALLS]\nsend({})\n\n[TOOL]\nsent\n"
    )
    out = strip_outputs(t)
    assert "read_file({})" in out and "send({})" in out and "pay the bill" in out
    assert "IBAN" not in out and "more" not in out and "sent" not in out
    assert out.count(f"[TOOL]\n{OMITTED}") == 2


def test_colon_turns_drop_multiline_outputs():
    head = "USER: do it\n\nASSISTANT: Tool calls:\n- bash\n\n"
    t = head + "TOOL: a\nERROR: b\n\nASSISTANT: done"
    assert strip_outputs(t) == head + f"TOOL: {OMITTED}\n\nASSISTANT: done"


def test_execution_info_is_dropped_but_code_kept():
    t = (
        "[ASSISTANT]\n\nCode:\n```\nprint(1)\n```\n\n"
        "Execution info:\nSTDOUT:\n1\n\n[ASSISTANT]\nnext"
    )
    out = strip_outputs(t)
    assert "print(1)" in out and "STDOUT" not in out
    assert out.endswith("[ASSISTANT]\nnext")


def test_strip_prompt_changes_only_the_envelope():
    p = "TOOL: keep<agent_trajectory>\nTOOL: drop\n</agent_trajectory>TOOL: keep"
    expected = (
        f"TOOL: keep<agent_trajectory>\nTOOL: {OMITTED}\n</agent_trajectory>TOOL: keep"
    )
    assert strip_prompt(p) == expected
    with pytest.raises(ValueError):
        strip_prompt("no envelope")
