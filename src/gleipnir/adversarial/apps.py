"""APPS rows: `[user]` problem, then `[assistant tool_call: submit]` code."""

from __future__ import annotations

from typing import NamedTuple

from gleipnir.adversarial.contract import join_prompt, split_prompt

USER = "[user]\n"
SUBMIT = "\n[assistant tool_call: submit]\n"


class Submission(NamedTuple):
    head: str
    problem: str
    code: str


def parse(prompt: str) -> Submission:
    head, trajectory = split_prompt(prompt)
    if not trajectory.startswith(USER) or trajectory.count(SUBMIT) != 1:
        raise ValueError("not an APPS submission trajectory")
    problem, code = trajectory[len(USER) :].split(SUBMIT)
    return Submission(head, problem, code)


def with_code(prompt: str, code: str) -> str:
    """The same row with its submitted code replaced, trailing newlines kept."""
    head, problem, original = parse(prompt)
    tail = original[len(original.rstrip("\n")) :]
    return join_prompt(head, f"{USER}{problem}{SUBMIT}{code.rstrip(chr(10))}{tail}")
