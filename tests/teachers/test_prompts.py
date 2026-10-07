from gleipnir.prompts import (
    build_student_prompt,
    format_student_target,
    parse_teacher_target,
    split_qwen_think_completion,
    truncate_text,
)


def test_prompt_separates_context_and_final_output() -> None:
    prompt = build_student_prompt(
        [
            {"role": "user", "content": "Is the switch safe?"},
            {"role": "assistant", "content": "Yes."},
        ],
        "Judge the behavior.",
        1_000,
        "tail",
        exclude_final_output_from_context=True,
    )
    context, output = prompt.split("\n\n<output>\n", maxsplit=1)
    assert "Is the switch safe?" in context
    assert "Yes." not in context
    assert output == "Yes.\n</output>"


def test_head_tail_truncation_preserves_both_ends() -> None:
    text = "start-" + "x" * 200 + "-end"
    truncated = truncate_text(text, 80, "head_tail")
    assert len(truncated) == 80
    assert truncated.startswith("start-")
    assert truncated.endswith("-end")
    assert "middle truncated" in truncated


def test_teacher_target_parser_uses_expected_label_fallback() -> None:
    raw = (
        "<reasoning_summary>The claim conflicts with the evidence.</reasoning_summary>"
    )
    assert parse_teacher_target(raw, expected_prediction=1) == (
        "The claim conflicts with the evidence.",
        1,
    )
    assert format_student_target("Evidence summary.", 0).endswith("Prediction:0")


def test_qwen_think_split_does_not_expose_unclosed_reasoning() -> None:
    assert split_qwen_think_completion("private reasoning only") is None
    assert split_qwen_think_completion("private</think>visible") == (
        "private",
        "visible",
    )
