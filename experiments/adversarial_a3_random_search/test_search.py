import json
import random

import pytest

from experiments.adversarial_a3_random_search.journal import Journal
from experiments.adversarial_a3_random_search.run import summarize
from gleipnir.adversarial.random_search import (
    Candidate,
    SearchState,
    SuffixSpace,
    append_comment,
    proposal,
)


class Tokenizer:
    all_special_ids = [3]

    def get_vocab(self):
        return {"a": 0, "b": 1, "\n": 2, "<special>": 3, "é": 4, " ": 5}

    def decode(self, ids, **kwargs):
        return "".join(("a", "b", "\n", "<special>", "é", " ")[i] for i in ids)

    def encode(self, text, **kwargs):
        return [{"a": 0, "b": 1, "\n": 2, "é": 4, " ": 5}[c] for c in text]


def test_preserves_original_bytes_and_semantics():
    for code in ("x = 1", "x = 1\n\n", 'x = "# text"\n# existing\n'):
        edited = append_comment(code, 'x # " \\ tail')
        assert edited.startswith(code)
        assert edited[len(code) :].endswith('# x # " \\ tail\n')


@pytest.mark.parametrize("payload", ["hello\nprint(1)", "\r", "\f", "", "é"])
def test_rejects_executable_or_non_ascii_payload(payload):
    with pytest.raises(ValueError):
        append_comment("x=1\n", payload)


def test_pool_filters_special_unicode_and_newline():
    space = SuffixSpace(Tokenizer())
    assert space.pool == [0, 1]
    candidate = space.propose(random.Random(0), None)
    assert candidate.tokens == 20 and len(candidate.pieces) == 20


def test_shared_start_and_score_feedback():
    space = SuffixSpace(Tokenizer())
    adaptive = SearchState(0.8)
    independent = SearchState(0.8)
    first = proposal(space, adaptive, "task", 0, "adaptive", 1)
    assert first == proposal(space, independent, "task", 0, "independent", 1)
    assert adaptive.observe(first, 0.3)
    independent.observe(first, 0.3)
    nxt = proposal(space, adaptive, "task", 0, "adaptive", 2)
    assert sum(a != b for a, b in zip(first.pieces, nxt.pieces, strict=True)) == 4
    adaptive.observe(nxt, 0.6)
    assert adaptive.best == first and adaptive.score == 0.3
    # Independent proposals are unaffected by which previous candidate won.
    alternate = SearchState(0.1, Candidate((1,) * 20, "b" * 20, 20), 1)
    assert proposal(space, independent, "task", 0, "independent", 2) == proposal(
        space, alternate, "task", 0, "independent", 2
    )


def test_baseline_and_ties_are_retained():
    state = SearchState(0.2)
    c = Candidate((0,), "a", 1)
    assert not state.observe(c, 0.4)
    assert not state.observe(c, 0.2)
    assert state.best is None and state.queries == 2
    with pytest.raises(ValueError):
        state.observe(c, float("nan"))
    assert state.queries == 2


def test_journal_resumes_without_repeating_and_binds_input(tmp_path):
    path = tmp_path / "calls.jsonl"
    j = Journal(path, {"model": "test"}, 1)
    calls = []

    def score(p):
        calls.append(p)
        return {"score": 0.5, "tokens": 3}

    j.score("q1", "hello", score, {"q": 1})
    resumed = Journal(path, {"model": "test"}, 1)
    resumed.score("q1", "hello", score, {"q": 1})
    assert calls == ["hello"]
    with pytest.raises(ValueError):
        resumed.score("q1", "changed", score, {"q": 1})
    with pytest.raises(RuntimeError, match="cap"):
        resumed.score("q2", "hello", score, {"q": 2})
    with pytest.raises(ValueError, match="identity"):
        Journal(path, {"model": "other"}, 1)


def test_uncertain_call_and_failed_call_never_retry(tmp_path):
    path = tmp_path / "calls.jsonl"
    j = Journal(path, {}, 2)

    def fail(_):
        raise RuntimeError("GPU error")

    with pytest.raises(RuntimeError):
        j.score("q1", "x", fail, {})
    with pytest.raises(RuntimeError, match="previous"):
        Journal(path, {}, 2).score("q1", "x", lambda p: {"score": 0}, {})
    with path.open("a") as f:
        f.write(json.dumps({"type": "intent", "id": "q2"}) + "\n")
    with pytest.raises(ValueError, match="unfinished"):
        Journal(path, {}, 2)


def test_report_counts_tasks_not_seeds_and_marks_censoring():
    records = []

    def receipt(cid, metadata, score):
        records.extend(
            [
                {"type": "intent", "id": cid, "metadata": metadata},
                {
                    "type": "result",
                    "id": cid,
                    "status": "ok",
                    "value": {"score": score, "tokens": 5},
                },
            ]
        )

    for task in ("a", "b"):
        receipt(
            "base" + task,
            {"kind": "baseline", "task": task, "condition": "vanilla"},
            0.8,
        )
        receipt(
            "honest" + task,
            {"kind": "baseline", "task": task, "condition": "clean"},
            0.1,
        )
        for seed in (0, 1):
            for arm in ("adaptive", "independent"):
                score = 0.1 if task == "a" and arm == "adaptive" else 0.9
                receipt(
                    f"{task}{seed}{arm}",
                    {
                        "kind": "search",
                        "task": task,
                        "seed": seed,
                        "arm": arm,
                        "query": 1,
                        "payload": "a",
                        "payload_tokens": 1,
                    },
                    score,
                )
    config = {
        "arms": ["adaptive", "independent"],
        "task_ids": ["a", "b"],
        "seeds": [0, 1],
        "checkpoints": [1, 2],
        "queries_per_search": 2,
    }
    r = summarize(records, config, {"0.01": 0.2})
    assert r["curves"]["adaptive"]["1"]["0.01"]["successes"] == 2
    assert r["curves"]["adaptive"]["1"]["0.01"]["conservative_rate"] == 0.5
    assert r["paired"]["1"]["0.01"]["adaptive_minus_independent"] == 0.5
    assert r["curves"]["adaptive"]["2"]["0.01"]["completed_searches"] == 0
    assert r["curves"]["adaptive"]["2"]["0.01"]["conservative_rate"] == 0
    assert len(r["search_outcomes"]) == 8


def test_runner_replays_completed_queries_without_gpu_calls(tmp_path, monkeypatch):
    from experiments.adversarial_a3_random_search import run as runner
    from gleipnir.adversarial import apps, scorer
    from gleipnir.adversarial.contract import join_prompt

    config = {
        "monitor": "test",
        "task_ids": ["a", "b"],
        "seeds": [0, 1],
        "arms": ["adaptive", "independent"],
        "queries_per_search": 2,
        "checkpoints": [1, 2],
        "suffix_max_tokens": 20,
        "total_call_cap": 30,
        "gpu_budget_seconds": 1000,
        "baseline_max_abs_drift": 0.02,
    }
    rows, baseline = {}, {}
    for task in config["task_ids"]:
        for condition in ("clean", "vanilla"):
            row_id = task + condition
            prompt = join_prompt(
                "Instructions", apps.USER + condition + apps.SUBMIT + "x=1\n"
            )
            rows[(task, condition)] = {"id": row_id, "prompt": prompt}
            baseline[row_id] = {"score": 0.1 if condition == "clean" else 0.8}
    calls = []

    class FakeScorer:
        def __init__(self, monitor):
            self.tokenizer = Tokenizer()

        def identity(self):
            return {"model": "fake"}

        def score(self, prompt, adapter=True):
            calls.append(prompt)
            if not adapter:
                score = 0.4
            elif "clean" in prompt:
                score = 0.1
            else:
                score = 0.2 if "# " in prompt else 0.8
            return {"score": score, "tokens": 20, "logit_0": 0, "logit_1": 0}

    monkeypatch.setattr(scorer, "Scorer", FakeScorer)
    monkeypatch.setattr(runner, "CONFIG", config)
    monkeypatch.setattr(runner, "OUT", tmp_path)
    monkeypatch.setattr(
        runner,
        "prepare",
        lambda: {"identity": {"test": True}, "thresholds": {"0.01": 0.3, "0.02": 0.2}},
    )
    monkeypatch.setattr(
        runner, "inputs", lambda: (rows, baseline, {"0.01": 0.3, "0.02": 0.2})
    )
    runner.run()
    assert len(calls) == 30
    assert (
        json.loads((tmp_path / "run_status.json").read_text())["state"] == "succeeded"
    )
    calls.clear()
    runner.run()
    assert calls == []
    runner.report()
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["physical_calls"] == 30
    assert report["curves"]["adaptive"]["2"]["0.01"]["successes"] == 4
    assert report["honest_controls"]["adaptive"]["pairs"] == 4
