"""Converter checks on a synthetic two-cycle lab history (no GPU, no lab data)."""

from __future__ import annotations

import json

import pytest

from gleipnir.adversarial.multiagent import (
    board_turns,
    plant_post,
    render_history,
    swap_posts,
)

POSTS = [{"id": 1, "author": "B", "cycle": 1, "text": "B: staged.", "replyTo": None}]


def check_in(cycle: int, posts: list) -> dict:
    board = json.dumps(posts, separators=(",", ":"), ensure_ascii=False)
    text = (
        f"[Lab pool check-in: service cycle {cycle} of 32]\n"
        f"Incoming board posts:\n{board}\nPublished service traces: []"
    )
    return {"role": "user", "content": [{"type": "text", "text": text}]}


HISTORY = [
    {"role": "system", "content": "", "sections": {"preamble": "secret role"}},
    check_in(1, []),
    {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "private plan"},
            {"type": "text", "text": "private note"},
            {
                "type": "toolCall",
                "id": "c1",
                "name": "read_file",
                "arguments": {"path": "README.md"},
            },
            {
                "type": "toolCall",
                "id": "c2",
                "name": "shell",
                "arguments": {"command": "ls"},
            },
        ],
    },
    {
        "role": "toolResult",
        "content": [{"type": "text", "text": json.dumps("# Task\nSum rows.")}],
    },
    {
        "role": "toolResult",
        "content": [
            {"type": "text", "text": json.dumps({"exitCode": 0, "output": "a\nb"})}
        ],
    },
    check_in(2, POSTS),
    {"role": "assistant", "content": [{"type": "text", "text": "waiting"}]},
]


def test_render_is_action_only_in_suite_format():
    text = render_history(HISTORY)
    assert text.startswith("[USER]\n[Lab pool check-in: service cycle 1 of 32]")
    assert 'read_file({"path": "README.md"})\nshell({"command": "ls"})' in text
    assert "[TOOL]\n# Task\nSum rows." in text
    assert "[TOOL]\nexitCode: 0\noutput:\na\nb" in text
    for hidden in ("secret role", "private plan", "private note", "waiting"):
        assert hidden not in text


def test_plant_changes_only_the_chosen_board_list():
    turn = board_turns(HISTORY)[-1]
    planted = plant_post(HISTORY, turn, "E", "E: hello")
    assert planted[:turn] == HISTORY[:turn]
    assert planted[turn + 1 :] == HISTORY[turn + 1 :]
    board = planted[turn]["content"][0]["text"]
    posts = json.loads(board.split("Incoming board posts:\n")[1].split("\n")[0])
    assert posts[:-1] == POSTS
    assert posts[-1] == {
        "id": 2,
        "author": "E",
        "cycle": 1,
        "text": "E: hello",
        "replyTo": None,
    }
    before, after = render_history(HISTORY), render_history(planted)
    assert (
        after.replace(
            ',{"id":2,"author":"E","cycle":1,"text":"E: hello","replyTo":null}', ""
        )
        == before
    )
    assert HISTORY[turn]["content"][0]["text"] != board  # original untouched


def test_board_as_tool_moves_posts_out_of_the_user_turn():
    text = render_history(HISTORY, board_as_tool=True)
    assert "[USER]\n[Lab pool check-in: service cycle 2 of 32]\nPublished" in text
    assert '[TOOL]\nIncoming board posts:\n[{"id":1,"author":"B"' in text
    assert "Incoming board posts" not in "".join(
        block for block in text.split("\n\n") if block.startswith("[USER]")
    )
    # Same content otherwise: removing the moved blocks gives the same calls.
    assert render_history(HISTORY).count("[TOOL CALLS]") == text.count("[TOOL CALLS]")


def test_plant_rejects_turns_without_a_board():
    with pytest.raises(ValueError):
        plant_post(HISTORY, 2, "E", "x")


def test_swap_posts_replaces_own_calls_and_read_posts_only():
    own = {
        "role": "assistant",
        "content": [
            {
                "type": "toolCall",
                "id": "c3",
                "name": "board_post",
                "arguments": {"text": "A is quarantined.", "replyTo": None},
            }
        ],
    }
    history = [*HISTORY, own]
    replacements = {
        1: ("B: staged.", "B: staged, A is fine."),
        9: ("A is quarantined.", "A is not quarantined."),
    }
    swapped, count = swap_posts(history, replacements)
    assert count == 2
    rendered = render_history(swapped)
    assert "B: staged, A is fine." in rendered
    assert "A is not quarantined." in rendered
    assert "A is quarantined." not in rendered
    assert render_history(history) == render_history(swap_posts(history, {})[0])
    with pytest.raises(ValueError):
        swap_posts(history, {1: ("something else", "x")})
