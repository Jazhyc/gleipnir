"""Prompt construction and teacher-output parsing for monitor distillation.

This module began as the reusable core of the Aletheia privileged-information
distillation track. It deliberately contains no competition runner, NDIF, or
model-serving dependency.
"""

from __future__ import annotations

import re
from typing import Any

SUMMARY_PATTERN = re.compile(
    r"<reasoning_summary>\s*(.*?)\s*</reasoning_summary>\s*Prediction:\s*([01])\b",
    flags=re.DOTALL,
)
SUMMARY_ONLY_PATTERN = re.compile(
    r"<reasoning_summary>\s*(.*?)\s*</reasoning_summary>",
    flags=re.DOTALL,
)
SUMMARY_RATING_PATTERN = re.compile(
    r"<reasoning_summary>\s*(.*?)\s*</reasoning_summary>\s*"
    r"Rating:\s*([1-7])\b\s*Prediction:\s*([01])\b",
    flags=re.DOTALL,
)
SUMMARY_RATING_ONLY_PATTERN = re.compile(
    r"<reasoning_summary>\s*(.*?)\s*</reasoning_summary>\s*"
    r"Rating:\s*([1-7])\b",
    flags=re.DOTALL,
)
COUNTERFACTUAL_PATTERN = re.compile(
    r"<reasoning_summary>\s*(.*?)\s*</reasoning_summary>\s*"
    r"<facts>\s*(.*?)\s*</facts>\s*"
    r"<contradiction>\s*(.*?)\s*</contradiction>\s*"
    r"Prediction:\s*([01])\b",
    flags=re.DOTALL,
)
COUNTERFACTUAL_ONLY_PATTERN = re.compile(
    r"<reasoning_summary>\s*(.*?)\s*</reasoning_summary>\s*"
    r"<facts>\s*(.*?)\s*</facts>\s*"
    r"<contradiction>\s*(.*?)\s*</contradiction>",
    flags=re.DOTALL,
)

REFERENCE_PREAMBLE = (
    "The following retrieved facts may be incomplete, ambiguous, or irrelevant. "
    "Use them only when they clearly concern a material claim in the output."
)


def safe_text(value: Any) -> str:
    return (
        str(value if value is not None else "")
        .encode("utf-8", errors="replace")
        .decode("utf-8", errors="replace")
    )


def merge_messages(messages: Any) -> list[dict[str, str]]:
    if isinstance(messages, dict):
        messages = [messages]
    if not isinstance(messages, list):
        messages = [{"role": "unknown", "content": messages}]
    merged: list[dict[str, str]] = []
    for item in messages:
        if not isinstance(item, dict):
            item = {"role": "unknown", "content": item}
        role = safe_text(item.get("role", "unknown"))
        content = safe_text(item.get("content", ""))
        if merged and merged[-1]["role"] == role:
            merged[-1]["content"] += "\n" + content
        else:
            merged.append({"role": role, "content": content})
    return merged


def final_assistant_reasoning(messages: Any) -> str:
    """Return the final assistant's optional reasoning trace."""
    if isinstance(messages, dict):
        messages = [messages]
    if not isinstance(messages, list):
        return ""
    for item in reversed(messages):
        if not isinstance(item, dict):
            continue
        if safe_text(item.get("role", "")) == "assistant":
            return safe_text(item.get("reasoning", ""))
    return ""


def truncate_text(text: str, max_chars: int, mode: str) -> str:
    if len(text) <= max_chars:
        return text
    if mode == "tail":
        return text[-max_chars:]
    if mode == "head":
        return text[:max_chars]
    if mode in {"head_tail", "head_tail_25", "head_tail_75"}:
        marker = "\n\n[... middle truncated ...]\n\n"
        available = max(0, max_chars - len(marker))
        head_fraction = {
            "head_tail": 0.5,
            "head_tail_25": 0.25,
            "head_tail_75": 0.75,
        }[mode]
        head = int(available * head_fraction)
        return text[:head] + marker + text[-(available - head) :]
    raise ValueError(f"unknown truncation mode: {mode}")


def format_example(
    messages: Any,
    max_chars: int,
    truncation: str,
    *,
    exclude_final_assistant_from_context: bool = False,
) -> tuple[str, str]:
    merged = merge_messages(messages)
    final_assistant_index = next(
        (
            index
            for index in range(len(merged) - 1, -1, -1)
            if merged[index]["role"] == "assistant"
        ),
        None,
    )
    output = (
        merged[final_assistant_index]["content"]
        if final_assistant_index is not None
        else ""
    )
    context_messages = merged
    if exclude_final_assistant_from_context and final_assistant_index is not None:
        context_messages = (
            merged[:final_assistant_index] + merged[final_assistant_index + 1 :]
        )
    context = "\n\n".join(
        f"{message['role'].upper()}: {message['content']}"
        for message in context_messages
    )
    return truncate_text(context, max_chars, truncation), output


def build_student_prompt(
    messages: Any,
    prompt_template: str,
    max_chars: int,
    truncation: str,
    *,
    include_reasoning: bool = False,
    reasoning_max_chars: int = 0,
    reasoning_truncation: str = "head_tail",
    exclude_final_output_from_context: bool = False,
) -> str:
    context, output = format_example(
        messages,
        max_chars,
        truncation,
        exclude_final_assistant_from_context=exclude_final_output_from_context,
    )
    prompt = (
        f"{prompt_template}\n\n<context>\n{context}\n</context>"
        f"\n\n<output>\n{output}\n</output>"
    )
    if include_reasoning and reasoning_max_chars > 0:
        reasoning = final_assistant_reasoning(messages)
        if reasoning:
            reasoning = truncate_text(
                reasoning, reasoning_max_chars, reasoning_truncation
            )
            prompt += f"\n\n<assistant_reasoning>\n{reasoning}\n</assistant_reasoning>"
    return prompt


def append_reference_material(student_prompt: str, reference: str) -> str:
    """Append a consistently delimited, explicitly untrusted retrieval block."""
    reference = str(reference).strip()
    if not reference:
        reference = "- No retrieved reference material."
    return (
        f"{student_prompt}\n\n<reference_material>\n"
        f"{REFERENCE_PREAMBLE}\n{reference}\n</reference_material>"
    )


def route_reference_material(
    student_prompt: str,
    reference: str,
    visibility: str,
) -> tuple[str, str]:
    """Return the deployable student prompt and the evidence-aware teacher prompt."""
    teacher_prompt = append_reference_material(student_prompt, reference)
    if visibility == "teacher_only":
        return student_prompt, teacher_prompt
    if visibility == "teacher_and_student":
        return teacher_prompt, teacher_prompt
    raise ValueError(
        "teacher.reference_visibility must be 'teacher_only' or "
        f"'teacher_and_student', got {visibility!r}"
    )


def build_teacher_prompt(student_prompt: str, teacher_template: str, label: int) -> str:
    privileged = teacher_template.replace("__GROUND_TRUTH__", str(int(label)))
    _, separator, evidence = student_prompt.partition("<context>")
    if not separator:
        raise ValueError("student prompt is missing <context>")
    return f"{privileged}\n\n<context>{evidence}"


def extract_harmony_final(raw_completion: str) -> str:
    """Return only GPT-OSS's visible final channel, excluding innate analysis."""
    text = safe_text(raw_completion)
    markers = ("assistantfinal", "<|channel|>final<|message|>")
    positions = [(text.rfind(marker), marker) for marker in markers]
    position, marker = max(positions, key=lambda item: item[0])
    return text[position + len(marker) :].strip() if position >= 0 else text.strip()


def split_qwen_think_completion(raw_completion: str) -> tuple[str, str] | None:
    """Split Qwen's private thinking from its visible answer.

    Qwen3.5's chat template ends the generation prompt with ``<think>``. The
    returned completion therefore normally contains the private continuation
    followed by ``</think>`` and the visible answer, without another opening
    marker. Missing closure is treated as malformed rather than exposing the
    entire completion as a student target.
    """
    text = safe_text(raw_completion)
    boundary = text.find("</think>")
    if boundary < 0:
        return None
    reasoning = text[:boundary]
    if reasoning.lstrip().startswith("<think>"):
        reasoning = reasoning.lstrip()[len("<think>") :]
    return reasoning.strip(), text[boundary + len("</think>") :].strip()


def extract_teacher_final(
    raw_completion: str,
    output_format: str = "harmony",
) -> str | None:
    """Extract only the teacher's visible answer for the configured model family."""
    if output_format == "harmony":
        return extract_harmony_final(raw_completion)
    if output_format == "qwen_think":
        split = split_qwen_think_completion(raw_completion)
        return split[1] if split is not None else None
    raise ValueError(f"unknown teacher output format: {output_format!r}")


def parse_teacher_target(
    raw_completion: str,
    expected_prediction: int | None = None,
    output_format: str = "harmony",
) -> tuple[str, int] | None:
    final = extract_teacher_final(raw_completion, output_format)
    if final is None:
        return None
    matches = list(SUMMARY_PATTERN.finditer(final))
    if matches:
        summary, prediction = matches[-1].groups()
        prediction = int(prediction)
    elif expected_prediction in (0, 1):
        summary_matches = list(SUMMARY_ONLY_PATTERN.finditer(final))
        if not summary_matches:
            return None
        summary = summary_matches[-1].group(1)
        prediction = int(expected_prediction)
    else:
        return None
    summary = " ".join(summary.split())
    if not summary:
        return None
    return summary, prediction


def parse_counterfactual_teacher_target(
    raw_completion: str,
    expected_prediction: int | None = None,
    output_format: str = "harmony",
) -> tuple[str, str, str, int] | None:
    """Parse the ordered summary, facts, contradiction, and prediction fields."""
    final = extract_teacher_final(raw_completion, output_format)
    if final is None:
        return None
    matches = list(COUNTERFACTUAL_PATTERN.finditer(final))
    if matches:
        summary, facts, contradiction, prediction = matches[-1].groups()
        prediction = int(prediction)
    elif expected_prediction in (0, 1):
        matches = list(COUNTERFACTUAL_ONLY_PATTERN.finditer(final))
        if not matches:
            return None
        summary, facts, contradiction = matches[-1].groups()
        prediction = int(expected_prediction)
    else:
        return None
    fields = tuple(" ".join(value.split()) for value in (summary, facts, contradiction))
    if not all(fields):
        return None
    return fields[0], fields[1], fields[2], prediction


def parse_rating_teacher_target(
    raw_completion: str,
    expected_prediction: int | None = None,
    output_format: str = "harmony",
) -> tuple[str, int, int] | None:
    """Parse a summary, signed 1--7 evidence rating, and binary prediction."""
    final = extract_teacher_final(raw_completion, output_format)
    if final is None:
        return None
    matches = list(SUMMARY_RATING_PATTERN.finditer(final))
    if matches:
        summary, rating, prediction = matches[-1].groups()
        rating, prediction = int(rating), int(prediction)
    elif expected_prediction in (0, 1):
        matches = list(SUMMARY_RATING_ONLY_PATTERN.finditer(final))
        if not matches:
            return None
        summary, rating = matches[-1].groups()
        rating, prediction = int(rating), int(expected_prediction)
    else:
        return None
    summary = " ".join(summary.split())
    if not summary:
        return None
    return summary, rating, prediction


def rating_matches_prediction(rating: int, prediction: int) -> bool:
    """Allow neutral 4 for either label and require signed polarity otherwise."""
    if not 1 <= rating <= 7 or prediction not in (0, 1):
        return False
    return rating <= 4 if prediction == 0 else rating >= 4


def format_student_target(
    summary: str,
    prediction: int,
    facts: str | None = None,
    contradiction: str | None = None,
    rating: int | None = None,
) -> str:
    if facts is not None and contradiction is not None:
        return (
            f"<reasoning_summary>\n{summary.strip()}\n</reasoning_summary>\n"
            f"<facts>\n{facts.strip()}\n</facts>\n"
            f"<contradiction>\n{contradiction.strip()}\n</contradiction>\n"
            f"Prediction:{int(prediction)}"
        )
    if rating is not None:
        if not 1 <= int(rating) <= 7:
            raise ValueError(f"rating must be between 1 and 7, got {rating}")
        return (
            f"<reasoning_summary>\n{summary.strip()}\n</reasoning_summary>\n"
            f"Rating:{int(rating)}\n"
            f"Prediction:{int(prediction)}"
        )
    return (
        f"<reasoning_summary>\n{summary.strip()}\n</reasoning_summary>\n"
        f"Prediction:{int(prediction)}"
    )
