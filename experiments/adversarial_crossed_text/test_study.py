"""Failure, cache identity, task pairing and inference-contract checks."""

import json
from unittest.mock import Mock

import pytest

from experiments.adversarial_crossed_text import common as c
from experiments.adversarial_crossed_text import evaluate as e
from experiments.adversarial_crossed_text import frontier as f
from experiments.adversarial_crossed_text import pipeline as p
from gleipnir.adversarial import apps
from gleipnir.adversarial.attacker import AttackerError


def test_real_split_has_100_unused_tasks_and_disjoint_calibration():
    cfg = c.config()
    old, triggers = c.prior(cfg)
    selected = p.select(cfg, old, triggers)
    dev, held, cal = [
        set(selected[k]) for k in ("development", "heldout", "calibration")
    ]
    assert (len(dev), len(held), len(cal)) == (50, 100, 221)
    assert not (dev & held or dev & cal or held & cal)
    assert not held & set(old["development"] + old["heldout"])


def test_prompt_substitution_never_replaces_user_braces():
    assert (
        p.fill("{problem} {text}", {"problem": "literal {text}", "text": "new"})
        == "literal {text} new"
    )


def test_mock_live_or_input_changes_cannot_reuse_cache(tmp_path):
    cfg = c.config()
    c.bind(tmp_path, cfg, "run", {"mock": True})
    c.bind(tmp_path, cfg, "run", {"mock": True})
    with pytest.raises(SystemExit):
        c.bind(tmp_path, cfg, "run", {"mock": False})
    cfg["texts"]["target"] += 1
    with pytest.raises(SystemExit):
        c.bind(tmp_path, cfg, "run", {"mock": True})


def test_reserved_tinker_call_is_not_redispatched_and_counts_budget(tmp_path):
    cfg = c.config()
    messages = [{"role": "user", "content": "hello"}]
    c.append(
        tmp_path / "calls.jsonl",
        {
            "request_id": "x",
            "role": "writer",
            "charged_usd": 0.1,
            "prompt_sha256": c.sha(json.dumps(messages, sort_keys=True)),
            "status": "reserved",
        },
    )
    calls = c.Calls(tmp_path, cfg, "writer", True, lambda _: "unused")
    calls.client.chat = Mock(side_effect=AssertionError("must not dispatch"))
    assert calls.chat("x", messages)["status"] == "interrupted_unknown"
    assert c.spent(tmp_path, "writer") == 0.1
    with pytest.raises(ValueError):
        calls.chat("x", [{"role": "user", "content": "changed"}])


def test_tinker_ambiguous_error_keeps_reservation_and_terminal_failure(tmp_path):
    cfg = c.config()
    calls = c.Calls(tmp_path, cfg, "writer", True, lambda _: "unused")
    calls.dry_run = False
    calls.client.chat = Mock(side_effect=AttackerError("timeout"))
    messages = [{"role": "user", "content": "hello"}]
    assert calls.chat("x", messages)["status"] == "infrastructure_failure"
    charge = c.spent(tmp_path, "writer")
    assert charge > 0
    calls.chat("x", messages)
    assert calls.client.chat.call_count == 1
    assert c.spent(tmp_path, "writer") == charge


def records(values, condition="none", monitor="a"):
    return [
        {
            "id": f"{task}:{code}:{condition}",
            "task_id": str(task),
            "code": code,
            "score": value,
            "status": "ok",
            "set": "main",
            "split": "heldout",
            "text": condition,
            "monitor": monitor,
        }
        for task, pair in enumerate(values)
        for code, value in zip(("honest", "backdoored"), pair, strict=True)
    ]


def test_complete_pairs_only_and_auc_ties_count_half():
    rows = records([[0.1, 0.9], [0.4, 0.4], [0.8, 0.2]])
    assert e.auc(list(e.pairs(rows, "none", "heldout").values())) == pytest.approx(
        11 / 18
    )
    assert e.auc([[0.4, 0.4], [0.4, 0.4]]) == 0.5
    rows[1]["status"] = "refusal"
    assert set(e.pairs(rows, "none", "heldout")) == {"1", "2"}


def test_monitor_difference_reuses_same_task_draws():
    a = {str(i): [0.1, 0.9] for i in range(5)}
    b = {str(i): [0.9, 0.1] for i in range(3)}
    result = e.paired_difference(a, b, 30, 0)
    assert result["n_tasks"] == 3
    assert result["difference"] == 1
    assert result["ci95"] == [1, 1]


def test_missing_score_bounds_cover_both_extremes():
    cfg = c.config()
    cfg["analysis"]["bootstrap_resamples"] = 20
    rows = records([[0.1, 0.9], [0.1, 0.9]])
    rows[0]["status"] = "refusal"
    rows += [{"id": "cal1", "status": "ok", "score": 0.2, "set": "calibration"}]
    variants = [
        {"id": r["id"], "metadata": {k: r[k] for k in ("split", "text", "code")}}
        for r in rows[:4]
    ]
    result = e.describe(rows, variants, cfg, "heldout")["conditions"]["none"]
    honest = result["flag_rates"]["0.01"]["honest"]
    assert honest["scored"] == 1 and honest["expected"] == 2
    assert honest["missing_inclusive_bounds"] == [0, 0.5]
    assert result["paired_tasks"] == 1


def test_frontier_cap_charges_unknown_reservations_and_blocks_repeat(tmp_path):
    cfg = c.config()
    cfg["protocol_status"] = "draft"
    cfg["frontier"]["caps_usd"]["openrouter"] = 0.2
    m = f.monitor("gpt-6.1-sol", cfg)
    reservation = f.reserve(tmp_path, cfg, m, "r1", 0.15, "prompt")
    f.settle(tmp_path, reservation, None)
    with pytest.raises(SystemExit):
        f.reserve(tmp_path, cfg, m, "r1", 0.01, "prompt")
    with pytest.raises(SystemExit):
        f.reserve(tmp_path, cfg, m, "r2", 0.06, "prompt")


def test_frontier_transport_failure_has_one_dispatch():
    cfg = c.config()
    m = f.monitor("kimi-k3", cfg)
    post = Mock(side_effect=TimeoutError("lost connection"))
    row = {"id": "r1", "prompt": "example", "metadata": {}}
    result = f.request_record(m, row, "secret-value", post)
    assert result["status"] == "infrastructure_failure"
    assert post.call_count == 1
    assert "secret-value" not in json.dumps(result)


def test_each_real_program_accepts_identical_header_without_changing_code():
    cfg = c.config()
    pairs, _ = c.load_pairs(cfg)
    for task, pair in pairs.items():
        if task != next(iter(pairs)):
            break
        for row in pair.values():
            original = apps.parse(row["prompt"]).code
            header, value = p.docstring(
                'Example documentation with a literal \\\\ and triple quotes: """.', 79
            )
            edited, at = p.insert_header(original, header)
            assert p.check_variant(original, edited, header, at, value).ok
            assert apps.parse(p.edited_prompt(row, edited)).code == edited


def test_paid_gate_refuses_draft_before_network_or_gpu():
    cfg = c.config()
    cfg["protocol_status"] = "draft"
    with pytest.raises(SystemExit):
        c.frozen(cfg)


def test_campaign_caps_span_replacement_directories(tmp_path, monkeypatch):
    from experiments.adversarial_crossed_text import campaign

    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    cfg = c.config()
    cfg["campaign"]["output"] = "shared"
    first = campaign.Campaign(cfg, tmp_path / "old")
    second = campaign.Campaign(cfg, tmp_path / "replacement")
    reservation = first.reserve("first", "tinker", "crossed-writer", 11.9, 12)
    first.settle(reservation, None)
    with pytest.raises(SystemExit):
        second.reserve("second", "tinker", "crossed-writer", 0.2, 12)


def test_structured_refusal_is_terminal_without_rewrite(tmp_path, monkeypatch):
    from experiments.adversarial_code_or_comments import llm

    data = {
        "choices": [
            {"message": {"content": "", "refusal": "declined"}, "finish_reason": "stop"}
        ],
        "usage": {"cost": 0.0},
    }
    monkeypatch.setattr(llm, "mock_post", lambda _: lambda body, key, timeout: data)
    calls = c.Calls(tmp_path, c.config(), "writer", True, lambda _: "unused")
    result = calls.chat("refusal", [{"role": "user", "content": "example"}])
    assert result["finish_reason"] == "refusal"
    assert p.parse_result(result)[0] == "refusal"


def test_authentication_failure_stops_further_dispatch(tmp_path):
    calls = c.Calls(tmp_path, c.config(), "writer", True, lambda _: "unused")
    calls.client.chat = Mock(side_effect=AttackerError("auth", {"http_status": 401}))
    calls.chat("first", [{"role": "user", "content": "example"}])
    with pytest.raises(SystemExit):
        calls.chat("second", [{"role": "user", "content": "example"}])
    assert calls.client.chat.call_count == 1


def test_mock_manifest_cannot_feed_live_stage_and_drift_is_detected(tmp_path):
    cfg = c.config()
    c.write_json(tmp_path / "selection.json", {"development": [], "heldout": []})
    c.write_jsonl(tmp_path / "variants.jsonl", [])
    c.bind(
        tmp_path,
        cfg,
        "pipeline",
        {"dry_run": True, "selection_sha256": c.digest(tmp_path / "selection.json")},
    )
    c.seal_variants(tmp_path, cfg, True)
    c.validate_variants(tmp_path, cfg, True)
    with pytest.raises(SystemExit):
        c.validate_variants(tmp_path, cfg, False)
    c.write_jsonl(tmp_path / "variants.jsonl", [{"id": "changed"}])
    with pytest.raises(SystemExit):
        c.validate_variants(tmp_path, cfg, True)


def test_frontier_response_preserves_condition_through_analysis():
    from gleipnir.adversarial.contract import join_prompt

    cfg = c.config()
    m = f.monitor("gpt-6.1-sol", cfg)
    returned = []
    for code, score in [("honest", 10), ("backdoored", 90)]:
        response = Mock()
        response.json.return_value = {
            "choices": [
                {"finish_reason": "stop", "message": {"content": f"Score:{score}"}}
            ],
            "usage": {"cost": 0.0},
        }
        row = {
            "id": code,
            "prompt": join_prompt(c.instruction("teacher", cfg), "example"),
            "metadata": {
                "task_id": "x",
                "code": code,
                "text": "none",
                "split": "heldout",
                "set": "main",
            },
        }
        record = f.request_record(m, row, "unused", Mock(return_value=response))
        assert record["text"] == "none"
        assert record["response_text"] == f"Score:{score}"
        returned.append(record)
    assert e.pairs(returned, "none", "heldout") == {"x": [0.1, 0.9]}
