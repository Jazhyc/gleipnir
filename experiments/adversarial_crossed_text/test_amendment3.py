"""Offline smoke decisions, dispatch barrier and pinned fallback restoration."""

import json
from types import SimpleNamespace

import pytest

from experiments.adversarial_crossed_text import amendment3 as a
from experiments.adversarial_crossed_text import common as c
from experiments.adversarial_crossed_text import pipeline as p
from experiments.adversarial_crossed_text.test_pipeline_guards import (
    study,
    refresh_reads,
)


def setup_smoke(study, monkeypatch, finished=5, valid=2):
    cfg = study.cfg
    cfg["amendment_2"] = {
        "max_attempts": 2,
        "writer_model_role": "cover_writer",
        "regenerate_conditions": ["cover_story"],
    }
    cfg["amendment_3"] = {
        "smoke": {
            "tasks": 5,
            "hash_seed": "frozen",
            "rule": "hash",
            "minimum_finished": 4,
            "minimum_hand_valid": 2,
        }
    }
    study.selection["development"] = study.tasks
    study.selection["heldout"] = []
    c.write_json(study.out / "selection.json", study.selection)
    tasks = a.smoke_tasks(study.out, cfg, study.selection, set(study.tasks))
    rows = c.read_jsonl(study.out / "texts.jsonl")
    for r in rows:
        if r["condition"] == "cover_story":
            r["attempt"] = 2
            r["status"] = "ok" if tasks.index(r["task_id"]) < finished else "truncation"
    c.write_jsonl(study.out / "texts.jsonl", rows)
    refresh_reads(study)
    hands = c.read_jsonl(study.out / "hand_audit.jsonl")
    for h in hands:
        if h["condition"] == "cover_story":
            h["valid"] = tasks.index(h["task_id"]) < valid
    c.write_jsonl(study.out / "hand_audit.jsonl", hands)
    c.write_jsonl(
        study.out / "calls.jsonl",
        [
            {
                "request_id": f"text:{t}:cover_story:2",
                "status": "settled",
                "result": {
                    "status": "ok",
                    "finish_reason": "stop" if i < finished else "length",
                },
            }
            for i, t in enumerate(tasks)
        ],
    )
    return tasks


@pytest.mark.parametrize(
    "finished,valid,branch",
    [(4, 2, "proceed"), (5, 1, "fallback"), (3, 3, "fallback"), (5, 5, "proceed")],
)
def test_gate_both_thresholds_and_history_validation(
    study, monkeypatch, finished, valid, branch
):
    setup_smoke(study, monkeypatch, finished, valid)
    gate = a.smoke_gate(study.out, study.cfg, study.pairs, study.triggers)
    assert (gate["finished"], gate["hand_valid"], gate["branch"]) == (
        finished,
        min(finished, valid),
        branch,
    )
    assert a.smoke_gate(study.out, study.cfg) == gate
    rows = c.read_jsonl(study.out / "hand_audit.jsonl")
    for r in rows:
        if r["condition"] == "cover_story":
            r["reason"] = "tampered"
    c.write_jsonl(study.out / "hand_audit.jsonl", rows)
    with pytest.raises(ValueError, match="evidence missing"):
        a.smoke_gate(study.out, study.cfg)


def test_no_gate_before_all_current_hand_reads(study, monkeypatch):
    setup_smoke(study, monkeypatch)
    rows = c.read_jsonl(study.out / "hand_audit.jsonl")
    c.write_jsonl(
        study.out / "hand_audit.jsonl",
        [r for r in rows if r["condition"] != "cover_story"],
    )
    with pytest.raises(SystemExit, match="hand-read all five"):
        a.smoke_gate(study.out, study.cfg, study.pairs, study.triggers)


def test_fallback_append_import_preserves_existing_and_is_idempotent(
    study, monkeypatch
):
    setup_smoke(study, monkeypatch, 3, 1)
    a.smoke_gate(study.out, study.cfg, study.pairs, study.triggers)
    study.cfg["amendment_3"]["fallback_import"] = {"source": "pinned-amendment1"}
    payloads = {}
    for name in ("texts.jsonl", "text_checks.jsonl", "hand_audit.jsonl"):
        cover = [
            r for r in c.read_jsonl(study.out / name) if r["condition"] == "cover_story"
        ]
        payloads[name] = "".join(json.dumps(r) + "\n" for r in cover).encode()
    monkeypatch.setattr(a, "fallback_payloads", lambda cfg: payloads)
    before = {n: (study.out / n).read_bytes() for n in payloads}
    a.restore_fallback(study.out, study.cfg)
    for n, data in payloads.items():
        assert (study.out / n).read_bytes() == before[n] + data
    a.restore_fallback(study.out, study.cfg)
    a.validate_fallback(study.out, study.cfg)
    for n, data in payloads.items():
        assert (study.out / n).read_bytes() == before[n] + data
    target = study.out / "texts.jsonl"
    target.write_bytes(target.read_bytes()[:-1] + b" \n")
    with pytest.raises(ValueError, match="artifact changed"):
        a.validate_fallback(study.out, study.cfg)


def test_passing_gate_forbids_fallback(study, monkeypatch):
    setup_smoke(study, monkeypatch)
    a.smoke_gate(study.out, study.cfg, study.pairs, study.triggers)
    with pytest.raises(SystemExit, match="failed text-validity"):
        a.restore_fallback(study.out, study.cfg)


def test_dispatch_only_smoke_and_failed_gate_blocks_all_generation(study, monkeypatch):
    tasks = setup_smoke(study, monkeypatch, 3, 1)
    a.smoke_gate(study.out, study.cfg, study.pairs, study.triggers)
    monkeypatch.setattr(p, "Calls", lambda *args, **kwargs: None)
    with pytest.raises(SystemExit, match="generation stopped"):
        p.generate(SimpleNamespace(dry_run=True, limit=None), study.out, study.cfg)
    (study.out / "smoke.gate.json").unlink()
    c.write_jsonl(
        study.out / "texts.jsonl",
        [
            r
            for r in c.read_jsonl(study.out / "texts.jsonl")
            if r["condition"] != "cover_story"
        ],
    )
    received = []

    class Calls:
        def chat(self, rid, messages, attempt):
            received.append(rid.split(":")[1])
            return {
                "status": "ok",
                "text": "Reasoned task requirement. " * 23,
                "finish_reason": "stop",
            }

    monkeypatch.setattr(p, "Calls", lambda *args, **kwargs: Calls())
    monkeypatch.setattr(
        p,
        "cover_round_bound",
        lambda out, cfg, ts, *args, **kwargs: received.append(tuple(ts)) or 1,
    )
    p.generate(SimpleNamespace(dry_run=True, limit=None), study.out, study.cfg)
    assert received[0] == tuple(tasks)
    assert set(received[1:]) == set(tasks)


def test_fallback_reads_pinned_source_and_preserves_verdicts(study):
    source = study.out
    c.write_json(
        source / "pipeline.identity.json",
        {"config_sha256": "source-config", "dry_run": False},
    )
    names = (
        "pipeline.identity.json",
        "texts.jsonl",
        "text_checks.jsonl",
        "hand_audit.jsonl",
    )
    study.cfg["amendment_3"] = {
        "fallback_import": {
            "source": str(source),
            "config_sha256": "source-config",
            "files": {n: c.digest(source / n) for n in names},
        }
    }
    payloads = a.fallback_payloads(study.cfg)
    for n in ("texts.jsonl", "text_checks.jsonl", "hand_audit.jsonl"):
        expected = [
            r for r in c.read_jsonl(source / n) if r["condition"] == "cover_story"
        ]
        assert [
            json.loads(line) for line in payloads[n].decode().splitlines()
        ] == expected
    with (source / "hand_audit.jsonl").open("a") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="fallback source artifact changed"):
        a.fallback_payloads(study.cfg)


def test_timeout_applies_only_to_cover_writer(study):
    cover = c.Calls(
        study.out, study.cfg, "writer", True, p.mock_reply, model_role="cover_writer"
    )
    checker = c.Calls(study.out, study.cfg, "checker", True, p.mock_reply)
    assert cover.client.timeout == 1800
    assert checker.client.timeout == 300


def test_passing_gate_continuation_preflights_whole_round_and_skips_smoke(
    study, monkeypatch
):
    tasks = setup_smoke(study, monkeypatch, 5, 2)
    a.smoke_gate(study.out, study.cfg, study.pairs, study.triggers)
    study.selection["heldout"] = ["x"]
    study.pairs["x"] = study.pairs["a"]
    study.triggers["x"] = {**study.triggers["a"], "task_id": "x"}
    c.write_json(study.out / "selection.json", study.selection)
    c.append(
        study.out / "trigger_audit.jsonl",
        {
            "task_id": "x",
            "status": "ok",
            "passed": True,
            **p.input_identity(study.pairs["x"], study.triggers["x"]),
        },
    )
    ref = p.final_texts(study.out)["a", "task_restatement"]
    c.append(study.out / "texts.jsonl", {**ref, "task_id": "x"})
    received, batches = [], []

    class Calls:
        def chat(self, rid, messages, attempt):
            received.append(rid)
            return {
                "status": "ok",
                "text": "Reasoned task requirement. " * 23,
                "finish_reason": "stop",
            }

    monkeypatch.setattr(p, "Calls", lambda *args, **kwargs: Calls())
    monkeypatch.setattr(
        p,
        "cover_round_bound",
        lambda out, cfg, ts, *args, **kwargs: batches.append(ts) or 1,
    )
    p.generate(SimpleNamespace(dry_run=True, limit=None), study.out, study.cfg)
    assert received == ["text:x:cover_story:1"]
    assert len(batches) == 1
    assert [t for batch in batches for t in batch] == [
        *study.selection["development"],
        "x",
    ]


def test_batch_over_budget_stops_before_dispatch(study, monkeypatch):
    setup_smoke(study, monkeypatch)
    received = []

    class Calls:
        def chat(self, *args):
            received.append(args)

    monkeypatch.setattr(p, "Calls", lambda *args, **kwargs: Calls())

    def over_cap(*args, **kwargs):
        raise SystemExit("entire remaining cover round cannot fit worst-case caps")

    monkeypatch.setattr(p, "cover_round_bound", over_cap)
    with pytest.raises(SystemExit, match="worst-case caps"):
        p.generate(SimpleNamespace(dry_run=True, limit=None), study.out, study.cfg)
    assert not received


def test_corrected_smoke_hand_verdict_forbids_continuation(study, monkeypatch):
    setup_smoke(study, monkeypatch, 5, 2)
    a.smoke_gate(study.out, study.cfg, study.pairs, study.triggers)
    hands = [
        r
        for r in c.read_jsonl(study.out / "hand_audit.jsonl")
        if r["condition"] == "cover_story"
    ]
    for row in hands:
        c.append(
            study.out / "hand_audit.jsonl",
            {**row, "valid": False, "reason": "Corrected after rereading."},
        )
    with pytest.raises(ValueError, match="superseded"):
        a.smoke_gate(study.out, study.cfg)


@pytest.mark.parametrize(
    "name", ["texts.jsonl", "hand_audit.jsonl", "text_checks.jsonl"]
)
def test_fallback_forbids_later_cover_overrides(study, monkeypatch, name):
    setup_smoke(study, monkeypatch, 3, 1)
    a.smoke_gate(study.out, study.cfg, study.pairs, study.triggers)
    study.cfg["amendment_3"]["fallback_import"] = {"source": "pinned"}
    payloads = {
        n: b"" for n in ("texts.jsonl", "hand_audit.jsonl", "text_checks.jsonl")
    }
    monkeypatch.setattr(a, "fallback_payloads", lambda cfg: payloads)
    a.restore_fallback(study.out, study.cfg)
    c.append(study.out / name, {"condition": "cover_story", "task_id": "replacement"})
    with pytest.raises(ValueError, match="overridden"):
        a.validate_fallback(study.out, study.cfg)


def test_partial_fallback_resumes_via_cli_without_parsing_partial_line(
    study, monkeypatch
):
    import sys

    setup_smoke(study, monkeypatch, 3, 1)
    a.smoke_gate(study.out, study.cfg, study.pairs, study.triggers)
    study.cfg["amendment_3"]["fallback_import"] = {"source": "pinned"}
    names = ("texts.jsonl", "hand_audit.jsonl", "text_checks.jsonl")
    payloads = {n: b'{"condition":"cover_story","task_id":"restore"}\n' for n in names}
    monkeypatch.setattr(a, "fallback_payloads", lambda cfg: payloads)
    before = {n: (study.out / n).read_bytes() for n in names}
    a.restore_fallback(study.out, study.cfg)
    (study.out / "texts.jsonl").write_bytes(
        before["texts.jsonl"] + payloads["texts.jsonl"][:9]
    )
    (study.out / "hand_audit.jsonl").write_bytes(before["hand_audit.jsonl"])
    monkeypatch.setattr(sys, "argv", ["pipeline", "restore-fallback"])
    monkeypatch.setattr(p, "config", lambda: study.cfg)
    monkeypatch.setattr(p, "output", lambda _: study.out)
    monkeypatch.setattr(p, "bind", lambda *args, **kwargs: None)

    def validate(*args, **kwargs):
        assert kwargs == {"allow_partial_fallback": True}

    monkeypatch.setattr(p, "validate_import", validate)
    p.main()
    a.validate_fallback(study.out, study.cfg)
    for n in names:
        assert (study.out / n).read_bytes() == before[n] + payloads[n]


def test_reserved_interrupted_smoke_is_unfinished_and_not_replayed(study, monkeypatch):
    tasks = setup_smoke(study, monkeypatch, 3, 1)
    target = tasks[-1]
    rows = c.read_jsonl(study.out / "texts.jsonl")
    for row in rows:
        if (row["task_id"], row["condition"]) == (target, "cover_story"):
            row["status"] = "interrupted_unknown"
    c.write_jsonl(study.out / "texts.jsonl", rows)
    refresh_reads(study)
    calls = c.read_jsonl(study.out / "calls.jsonl")
    for row in calls:
        if row["request_id"] == f"text:{target}:cover_story:2":
            row.pop("result")
            row["status"] = "reserved"
            row["charged_usd"] = 0.3
    c.write_jsonl(study.out / "calls.jsonl", calls)
    gate = a.smoke_gate(study.out, study.cfg, study.pairs, study.triggers)
    assert gate["finished"] == 3
    assert gate["branch"] == "fallback"
    assert c.read_jsonl(study.out / "calls.jsonl") == calls
