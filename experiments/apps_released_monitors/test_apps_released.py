from types import SimpleNamespace

import pytest

from experiments.apps_released_monitors.run import trajectory


def template(prefix: str) -> SimpleNamespace:
    close = "</agent_trajectory>"

    def render(body: str) -> str:
        separator = "" if body.endswith("\n") else "\n"
        return f"{prefix}{body}{separator}{close}\n"

    return SimpleNamespace(cache_prefix=prefix, trajectory_close=close, render=render)


def test_trajectory_round_trips_and_rewraps() -> None:
    student = template("short\n<agent_trajectory>\n")
    teacher = template("long policy\n<agent_trajectory>\n")
    prompt = student.render("[user]\nsolve it\n[assistant]\ncode")
    body = trajectory(prompt, student)
    assert body == "[user]\nsolve it\n[assistant]\ncode\n"
    assert (
        teacher.render(body)
        == f"long policy\n<agent_trajectory>\n{body}</agent_trajectory>\n"
    )


def test_trajectory_rejects_other_templates() -> None:
    with pytest.raises(ValueError):
        trajectory("other\n<agent_trajectory>\nx\n</agent_trajectory>\n", template("a"))
