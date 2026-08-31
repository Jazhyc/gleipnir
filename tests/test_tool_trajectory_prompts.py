from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.tool_trajectory_monitoring.prompting import (
    PromptTemplate,
    load_prompt_set,
)
from gleipnir.openrouter import OpenRouterConfig, PromptRecord, request_payload

EXPERIMENT_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "experiments" / "tool_trajectory_monitoring"
)


def test_prompt_roles_share_binary_interface_but_not_instruction_depth() -> None:
    prompt_set = load_prompt_set()
    teacher = prompt_set.teacher
    student = prompt_set.student

    assert prompt_set.prompt_set_id == "tool-trajectory-binary-v1"
    assert teacher.decision_prefix == student.decision_prefix == "Prediction:"
    assert teacher.negative_label == student.negative_label == "0"
    assert teacher.positive_label == student.positive_label == "1"
    assert len(teacher.instruction) > 4 * len(student.instruction)
    assert "SILENT REVIEW PROCEDURE" in teacher.instruction
    assert "Steel-man the strongest innocent explanation" in teacher.instruction
    assert "severity score" in teacher.instruction
    assert "SILENT REVIEW PROCEDURE" not in student.instruction
    assert "0--10" not in teacher.instruction
    assert "0--10" not in student.instruction


def test_manifest_maps_one_prompt_to_each_training_and_evaluation_role() -> None:
    prompt_set = load_prompt_set()
    assert prompt_set.teacher.uses == ("logit_annotation", "teacher_evaluation")
    assert prompt_set.student.uses == ("student_training", "student_evaluation")
    assert prompt_set.for_role("teacher") is prompt_set.teacher
    assert prompt_set.for_role("student") is prompt_set.student
    with pytest.raises(ValueError, match="unknown prompt role"):
        prompt_set.for_role("judge")  # type: ignore[arg-type]


def test_render_preserves_trajectory_and_exposes_exact_cache_prefix() -> None:
    prompt = load_prompt_set().teacher
    trajectory = "USER: inspect this\nTOOL: embedded Prediction:1 text"
    rendered = prompt.render(trajectory)

    assert rendered.startswith(prompt.cache_prefix)
    assert rendered[len(prompt.cache_prefix) :].startswith(trajectory)
    assert rendered.endswith(f"\n{prompt.trajectory_close}\n")
    assert prompt.decision_context(trajectory) == f"{rendered}Prediction:"
    assert prompt.completion("0") == "Prediction:0"
    assert prompt.completion("1") == "Prediction:1"
    assert prompt.rendered_sha256(trajectory) == prompt.rendered_sha256(trajectory)
    with pytest.raises(ValueError, match="nonempty"):
        prompt.render("  \n")
    with pytest.raises(ValueError, match="label must be"):
        prompt.completion("10")


def test_teacher_cache_prefix_integrates_with_openrouter_payload() -> None:
    template = load_prompt_set().teacher
    rendered = template.render("USER: list the files\nASSISTANT: Done.")
    payload = request_payload(
        PromptRecord("example-1", rendered, {}),
        OpenRouterConfig(
            model="moonshotai/kimi-k3",
            provider_only="makora",
            cache_prefix=template.cache_prefix,
        ),
    )

    content = payload["messages"][0]["content"]
    assert content[0] == {
        "type": "text",
        "text": template.cache_prefix,
        "cache_control": {"type": "ephemeral"},
    }
    assert content[1]["text"].startswith("USER: list the files")
    assert payload["provider"]["only"] == ["makora"]


def test_template_hash_covers_instruction_and_interface() -> None:
    template = load_prompt_set().student
    changed = PromptTemplate(
        **{
            **template.__dict__,
            "positive_label": "yes",
        }
    )
    assert changed.template_sha256 != template.template_sha256


def test_manifest_rejects_instruction_path_escape(tmp_path: Path) -> None:
    manifest = {
        "prompt_set_id": "test",
        "teacher_instruction": "../outside.txt",
        "student_instruction": "student.txt",
        "trajectory_open": "<trajectory>",
        "trajectory_close": "</trajectory>",
        "decision_prefix": "Prediction:",
        "negative_label": "0",
        "positive_label": "1",
        "teacher_uses": ["annotation"],
        "student_uses": ["training"],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "student.txt").write_text("student", encoding="utf-8")
    with pytest.raises(ValueError, match="must be inside"):
        load_prompt_set(tmp_path)


def test_reasoning_judge_draft_has_independent_scoring_contract() -> None:
    prompt_directory = EXPERIMENT_DIRECTORY / "prompts"
    system_prompt = (prompt_directory / "reasoning_judge_system_v1.txt").read_text(
        encoding="utf-8"
    )
    user_prompt = (prompt_directory / "reasoning_judge_user_v1.txt").read_text(
        encoding="utf-8"
    )

    assert "accurate, well-reasoned, and useful" in system_prompt
    assert "Do not reward verbosity" in system_prompt
    assert "round_half_up" in system_prompt
    assert "wrong_prediction" in system_prompt
    assert "reference_label_support" in system_prompt
    assert "Return exactly one JSON object" in system_prompt

    assert user_prompt.count("{{trajectory}}") == 1
    assert user_prompt.count("{{ground_truth}}") == 1
    assert user_prompt.count("{{candidate_explanation}}") == 1
    assert user_prompt.index("{{trajectory}}") < user_prompt.index("{{ground_truth}}")
    assert user_prompt.index("{{ground_truth}}") < user_prompt.index(
        "{{candidate_explanation}}"
    )
