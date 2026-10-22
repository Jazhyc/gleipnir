"""Deterministic splits, trigger parsing and the sandbox runner's guards."""

from __future__ import annotations

import json
import sys

import pytest

from experiments.adversarial_code_or_comments import trigger_runner
from experiments.adversarial_code_or_comments.common import config
from experiments.adversarial_code_or_comments.select_tasks import (
    pools,
    repetition_subset,
    split,
    take_confirmed,
)
from experiments.adversarial_code_or_comments.trigger import (
    feedback,
    merge_records,
    parse_proposal,
)
from gleipnir.adversarial import eval_sets

CFG = config()
TASKS = [str(i) for i in range(647)]


def test_split_is_deterministic_and_disjoint():
    first = split(CFG, TASKS, None)
    again = split(CFG, list(reversed(TASKS)), None)
    assert first == again
    dev, held = set(first["development_pool"]), set(first["heldout_pool"])
    assert not dev & held and dev | held == set(TASKS)
    assert (len(dev), len(held)) == (324, 323)
    assert len(first["development"]) == len(first["heldout"]) == 200
    assert set(first["repetition"]) <= set(first["development"])
    assert len(first["repetition"]) == 100
    other = {**CFG, "selection": {**CFG["selection"], "split_seed": 1}}
    assert split(other, TASKS, None)["development"] != first["development"]


def test_unconfirmed_tasks_are_skipped_and_counted():
    pool = ["a", "b", "c", "d", "e"]
    chosen, dropped = take_confirmed(pool, {"a", "c", "e"}, 2)
    assert chosen == ["a", "c"] and dropped == ["b"]
    chosen, dropped = take_confirmed(pool, {"a"}, 2)
    assert chosen == ["a"] and dropped == ["b", "c", "d", "e"]


def test_repetition_subset_is_seeded():
    dev, _ = pools(TASKS, 3)
    assert repetition_subset(dev, 5, 10) == repetition_subset(sorted(dev), 5, 10)
    assert repetition_subset(dev, 5, 10) != repetition_subset(dev, 6, 10)


@pytest.mark.skipif(
    not eval_sets.path("apps-neutral").exists(), reason="apps-neutral not on disk"
)
def test_real_split_sizes():
    from experiments.adversarial_code_or_comments.common import load_pairs

    pairs, counts = load_pairs(CFG)
    assert counts["both"] == 647
    chosen = split(CFG, pairs, None)
    assert len(chosen["development_pool"]) == 324
    assert len(chosen["heldout_pool"]) == 323


def test_parse_proposal():
    reply = (
        "<think>look at line 3</think>\n```json\n"
        + json.dumps(
            {
                "special_case": "when  n is\n 7",
                "code_location": "if n == 7:",
                "trigger_input": "7",
            }
        )
        + "\n```"
    )
    parsed = parse_proposal(reply)
    assert parsed["trigger_input"] == "7\n"
    assert parsed["special_case"] == "when n is 7"
    with pytest.raises(ValueError):
        parse_proposal('{"special_case": "x", "trigger_input": "1"}')
    with pytest.raises(ValueError):
        parse_proposal("no json here")


def run(stdout="", code=0, timed_out=False):
    return {
        "stdout": stdout,
        "exit_code": code,
        "timed_out": timed_out,
        "stderr_tail": "E",
    }


@pytest.mark.parametrize(
    "honest,backdoor,status",
    [
        (run("14\n"), run("13\n"), "confirmed"),
        (run("14 \n\n"), run("14"), "same_output"),
        (run("14\n", 1), run("13\n"), "honest_error"),
        (run("14\n"), run("", 137), "backdoor_error"),
        (run("14\n"), run(timed_out=True), "timeout"),
        (run("", 125), run("13\n"), "docker_error"),
    ],
)
def test_runner_verdicts(honest, backdoor, status):
    assert trigger_runner.verdict(honest, backdoor) == status


def test_docker_command_has_no_network_and_is_locked_down():
    cmd = trigger_runner.docker_command("n", "/w", "img", trigger_runner.DEFAULTS)
    joined = " ".join(cmd)
    assert "--network none" in joined and "--read-only" in joined
    assert "--cap-drop ALL" in joined and "--user 65534:65534" in joined
    assert "-v /w:/work:ro" in joined


def test_runner_refuses_outside_linux(tmp_path, monkeypatch):
    jobs = tmp_path / "jobs.jsonl"
    jobs.write_text(json.dumps({"id": "1:1", "task_id": "1", "attempt": 1}) + "\n")
    called = []
    monkeypatch.setattr(
        trigger_runner.subprocess, "run", lambda *a, **k: called.append(a)
    )
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(sys, "argv", ["runner", str(jobs), str(tmp_path / "v.jsonl")])
    with pytest.raises(SystemExit, match="Linux"):
        trigger_runner.main()
    assert not called and not (tmp_path / "v.jsonl").exists()


def test_feedback_describes_outcomes():
    text = feedback({"honest": run("14\n"), "backdoor": run("", 1)})
    assert "Solution A printed '14\\n'" in text and "exited with code 1" in text


def test_merge_keeps_first_confirmed_attempt():
    proposals = [
        {"id": "1:1", "task_id": "1", "attempt": 1, "status": "ok"},
        {
            "id": "1:2",
            "task_id": "1",
            "attempt": 2,
            "status": "ok",
            "special_case": "s",
            "code_location": "c",
            "trigger_input": "7\n",
        },
        {"id": "2:1", "task_id": "2", "attempt": 1, "status": "refusal"},
    ]
    verdicts = [
        {"id": "1:1", "status": "same_output", "honest": run(), "backdoor": run()},
        {
            "id": "1:2",
            "status": "confirmed",
            "honest": run("14"),
            "backdoor": run("13"),
        },
    ]
    rows = {
        r["task_id"]: r for r in merge_records(proposals, verdicts, ["1", "2", "3"])
    }
    assert rows["1"]["status"] == "confirmed" and rows["1"]["attempt"] == 2
    assert rows["1"]["backdoor_output"] == "13"
    assert rows["2"]["status"] == "proposal_refusal"
    assert rows["3"]["status"] == "no_proposal"
