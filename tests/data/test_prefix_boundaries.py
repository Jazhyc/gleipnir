import pytest

from gleipnir.prefix_boundaries import audit_boundaries


def test_statements_retained_without_targets_and_final_excluded():
    text = (
        "[USER]\ntask\n[ASSISTANT]\nstatement\n"
        "[ASSISTANT]\n[TOOL CALLS]\nf({})\n[TOOL]\nresult\n"
        "[ASSISTANT]\nfinal statement\n"
    )
    result = audit_boundaries(text, "stride")
    assert result.assistant_turns == 3
    assert result.statement_only_turns == 2
    assert len(result.candidate_ends) == 1
    prefix = text[: result.candidate_ends[0]]
    assert "statement" in prefix and "result" in prefix
    assert "final statement" not in prefix


def test_tool_output_cannot_create_tool_call_in_statement():
    text = "[ASSISTANT]\nhello\n[TOOL]\n[TOOL CALLS]\nf({})\n[ASSISTANT]\nend"
    assert not audit_boundaries(text, "gloom").candidate_ends


def test_final_tool_turn_excluded():
    assert not audit_boundaries(
        "[ASSISTANT]\n[TOOL CALLS]\nf({})", "bash_bench"
    ).candidate_ends


def test_bash_arena_boundary_and_malformed_warning():
    text = (
        "<step_1>\n<assistant>\n<tool_0>\nf\n</tool_0>\n</assistant>\n"
        "<tool_output_0>ok</tool_output_0>\n</step_1>\n"
        "<step_2>\n<assistant>\nend\n</step_2>\n"
    )
    result = audit_boundaries(text, "bash_arena")
    assert result.candidate_ends == (text.index("</step_1>") + len("</step_1>"),)
    assert result.warnings == ("assistant_delimiters_turn_2",)


def test_unknown_source_fails_closed():
    with pytest.raises(ValueError):
        audit_boundaries("text", "unknown")


def test_middle_summary_does_not_leak_into_preceding_action():
    text = (
        "<step_1>\n<assistant>\n<tool_0>\nf\n</tool_0>\n</assistant>\n"
        "</step_1>\n<middle_summary>future events</middle_summary>\n"
        "<step_8>\n<assistant>\nend\n</assistant>\n</step_8>\n"
    )
    result = audit_boundaries(text, "bash_arena")
    assert len(result.candidate_ends) == 1
    assert "future events" not in text[: result.candidate_ends[0]]


def test_missing_step_close_has_no_guessed_target():
    text = (
        "<step_1>\n<assistant>\n<tool_0>\nf\n</tool_0>\n</assistant>\n"
        "<step_2>\n<assistant>\nend\n</assistant>\n</step_2>\n"
    )
    result = audit_boundaries(text, "bash_arena")
    assert not result.candidate_ends
    assert "missing_step_close_turn_1" in result.warnings
