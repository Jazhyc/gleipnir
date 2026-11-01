"""Budget and API identity guards, without paid calls or credentials."""

import copy

import pytest

from . import frontier as f
from .common import config, digest
from .pipeline import validate_scores


def plan():
    spec, _ = f.settings()
    return {
        "settings": spec,
        "monitors": {
            n: {
                "identity": {"settings": s, "endpoint": spec["endpoints"][s["api"]]},
                "requests": [{"id": n}],
                "worst_case_usd": 0.5,
            }
            for n, s in spec["monitors"].items()
        },
    }


class Ledger:
    cfg = config()

    def __init__(self, records=None):
        self.records = records or {}

    def latest(self):
        return self.records


def test_full_bound_is_diagnostic_and_campaign_unknown_reservations_count():
    p = plan()
    p["monitors"]["gpt-6.1-sol"]["worst_case_usd"] = 10
    assert f.preflight(p, Ledger())["status"] == "per_request_reservations"
    ledger = Ledger(
        {
            "unknown": {
                "api": "openrouter",
                "charge": 9,
                "status": "reserved",
                "group": "another-study",
            }
        }
    )
    assert "openrouter shared campaign cap exceeded" in f.budget_reasons(
        p, ledger, "openrouter", 0.6
    )
    assert "openrouter A4 cap exceeded" in f.budget_reasons(
        p, Ledger(), "openrouter", 3.01
    )


def setup_run(tmp_path, monkeypatch, n=3, name="kimi-k3"):
    p = plan()
    for target in p["monitors"].values():
        target["requests"] = []
    p["monitors"][name]["requests"] = [
        {
            "id": str(i),
            "prompt": f.join_prompt(f.instruction(config(), "teacher"), str(i)),
            "metadata": {},
        }
        for i in range(n)
    ]
    cfg = copy.deepcopy(config())
    cfg["campaign"]["output"] = str(tmp_path / "campaign")
    monkeypatch.setattr(f, "committed", lambda *a: None)
    return p, cfg, f.settings()[1]


def test_next_request_over_cap_stops_before_key_or_reservation(tmp_path, monkeypatch):
    p, cfg, cc = setup_run(tmp_path, monkeypatch)
    monkeypatch.setattr(f, "request_bound", lambda *a: 3.01)
    monkeypatch.setattr(f, "load_key", lambda *a: pytest.fail("key loaded"))
    f.run(p, tmp_path, cfg, cc, "kimi-k3")
    assert not list((tmp_path / "units").rglob("*.intent.json"))
    assert not f.Campaign(cfg, tmp_path).latest()
    assert (tmp_path / "kimi-k3.budget_stop.json").exists()


def test_each_request_settles_before_next_reservation_and_resumes_without_retry(
    tmp_path, monkeypatch
):
    p, cfg, cc = setup_run(tmp_path, monkeypatch)
    monkeypatch.setattr(f, "request_bound", lambda *a: 2.0)
    monkeypatch.setattr(f, "load_key", lambda *a: "mock")
    calls = []

    def respond(m, row, *args):
        calls.append(row["id"])
        return {
            "id": row["id"],
            "status": "ok",
            "score": 0.2,
            "cost": 0.1,
            "model": m.model,
            "provider": "makora",
        }

    monkeypatch.setattr(f, "request_record", respond)
    f.run(p, tmp_path, cfg, cc, "kimi-k3")
    f.run(p, tmp_path, cfg, cc, "kimi-k3")
    assert calls == ["0", "1", "2"]
    assert sum(
        r["charge"] for r in f.Campaign(cfg, tmp_path).latest().values()
    ) == pytest.approx(0.3)


def test_unknown_cost_keeps_reservation_and_stops_next_row(tmp_path, monkeypatch):
    p, cfg, cc = setup_run(tmp_path, monkeypatch)
    monkeypatch.setattr(f, "request_bound", lambda *a: 2.0)
    monkeypatch.setattr(f, "load_key", lambda *a: "mock")
    monkeypatch.setattr(
        f,
        "request_record",
        lambda m, row, *a: {"id": row["id"], "status": "infrastructure_failure"},
    )
    f.run(p, tmp_path, cfg, cc, "kimi-k3")
    assert len(f.read(tmp_path / "kimi-k3.jsonl")) == 1
    assert sum(r["charge"] for r in f.Campaign(cfg, tmp_path).latest().values()) == 2


def test_ambiguous_transport_is_never_redispatched(tmp_path, monkeypatch):
    p, cfg, cc = setup_run(tmp_path, monkeypatch)
    monkeypatch.setattr(f, "load_key", lambda *a: "mock")

    def crash(*a):
        raise RuntimeError("crash")

    monkeypatch.setattr(f, "request_record", crash)
    with pytest.raises(RuntimeError, match="crash"):
        f.run(p, tmp_path, cfg, cc, "kimi-k3")
    with pytest.raises(SystemExit, match="ambiguous"):
        f.run(p, tmp_path, cfg, cc, "kimi-k3")
    assert len(f.Campaign(cfg, tmp_path).latest()) == 1


def test_saved_response_is_settled_on_resume(tmp_path, monkeypatch):
    p, cfg, cc = setup_run(tmp_path, monkeypatch, n=1)
    monkeypatch.setattr(f, "load_key", lambda *a: "mock")

    def save_and_crash(m, row, *a):
        f.append(
            tmp_path / "kimi-k3.jsonl",
            {
                "id": row["id"],
                "status": "ok",
                "model": m.model,
                "provider": "makora",
                "cost": 0.001,
            },
        )
        raise RuntimeError("crash after saving")

    monkeypatch.setattr(f, "request_record", save_and_crash)
    with pytest.raises(RuntimeError):
        f.run(p, tmp_path, cfg, cc, "kimi-k3")
    f.run(p, tmp_path, cfg, cc, "kimi-k3")
    assert (
        sum(r["charge"] for r in f.Campaign(cfg, tmp_path).latest().values()) == 0.001
    )


@pytest.mark.parametrize("status", ["ok", "refusal", "invalid_score", "batch_errored"])
def test_small_batches_collect_and_settle_only_their_own_rows(
    tmp_path, monkeypatch, status
):
    import sys
    from types import SimpleNamespace as NS

    name = "claude-sonnet-5.5"
    p, cfg, cc = setup_run(tmp_path, monkeypatch, n=41, name=name)
    monkeypatch.setattr(f, "load_key", lambda *a: "mock")
    monkeypatch.setattr(f, "request_bound", lambda *a: 0.08)
    batches = []

    class Batches:
        def create(self, requests):
            batches.append(requests)
            return NS(id=str(len(batches)))

        def retrieve(self, bid):
            return NS(processing_status="ended")

        def results(self, bid):
            return [
                NS(
                    custom_id=r["custom_id"],
                    result=NS(type="succeeded", message=NS(model="claude-sonnet-5-5")),
                )
                for r in batches[int(bid) - 1]
            ]

    monkeypatch.setitem(
        sys.modules,
        "anthropic",
        NS(Anthropic=lambda **k: NS(messages=NS(batches=Batches()))),
    )
    monkeypatch.setattr(
        f,
        "anthropic_result_record",
        lambda *a: {"status": status, "cost": 0.01, "score": 0.2, "error": "example"},
    )
    while len(f.read(tmp_path / f"{name}.jsonl")) < 41:
        f.run(p, tmp_path, cfg, cc, name)
        with pytest.raises(SystemExit, match="collect first"):
            f.run(p, tmp_path, cfg, cc, name)
        f.collect(p, tmp_path, cfg, cc)
    assert [len(b) for b in batches] == [18, 16, 7]
    assert sum(
        r["charge"] for r in f.Campaign(cfg, tmp_path).latest().values()
    ) == pytest.approx(0.41)


def test_frontier_receipt_requires_request_contract_instead_of_gpu_canary():
    identity = {
        "monitor": "kimi-k3",
        "backend": "frontier_api",
        "calibration_artifact": {"sha256": "pinned"},
    }
    identity["runtime_receipt_sha256"] = digest(identity)
    batch = {
        "monitor_identity": identity,
        "jobs": [{"id": "one", "prompt_sha256": "prompt"}],
    }
    receipt = {
        "batch_sha256": digest(batch),
        "monitor_identity": identity,
        "request_contract_sha256": digest(identity),
        "calibration_artifact": identity["calibration_artifact"],
        "rows": [
            {"id": "one", "prompt_sha256": "prompt", "status": "ok", "score": 0.2}
        ],
    }
    assert validate_scores(batch, receipt)["one"]["score"] == 0.2
    bad = copy.deepcopy(receipt)
    bad["calibration_artifact"]["sha256"] = "changed"
    # Deepcopy preserves the shared identity object, so explicitly separate it.
    bad["monitor_identity"] = identity
    with pytest.raises(ValueError, match="calibration"):
        validate_scores(batch, bad)
    receipt.pop("request_contract_sha256")
    receipt["canary_passed"] = True
    with pytest.raises(ValueError, match="contract"):
        validate_scores(batch, receipt)


def test_model_and_pinned_provider_drift_are_terminal():
    identity = {"settings": {"model": "moonshotai/kimi-k3", "provider_only": "makora"}}
    f.check_identity({"model": "moonshotai/kimi-k3", "provider": "Makora"}, identity)
    for row in (
        {"model": "other", "provider": "Makora"},
        {"model": "moonshotai/kimi-k3", "provider": "other"},
    ):
        with pytest.raises(ValueError, match="drift"):
            f.check_identity(row, identity)


def test_calibration_overlap_and_short_subset_are_rejected(monkeypatch):
    rows = [{"task_id": str(i)} for i in range(200)]
    monkeypatch.setattr(f.crossed, "calibration_records", lambda *a: rows)
    with pytest.raises(ValueError, match="overlaps"):
        f.calibration("kimi-k3", {}, {"output": "ignored"}, {"5"})
    rows.pop()
    with pytest.raises(ValueError, match="200-row"):
        f.calibration("kimi-k3", {}, {"output": "ignored"}, set())


def test_plan_deduplicates_across_arms_and_control_views(tmp_path, monkeypatch):
    from gleipnir.adversarial import apps
    from gleipnir.adversarial.contract import join_prompt

    spec, crossed_cfg = f.settings()
    prompt = join_prompt(
        f.instruction(config(), "teacher"),
        apps.USER + "Print one.\n" + apps.SUBMIT + "print(1)\n",
    )
    raw = {
        str(i): {"honest": {"prompt": prompt}, "backdoored": {"prompt": prompt}}
        for i in range(201)
    }
    rows = [
        f.job("200:honest:original", "200", "original", prompt, code="honest"),
        f.job("200:honest:winner", "200", "winner", prompt, code="honest"),
    ]
    monkeypatch.setattr(f, "benchmark_pairs", lambda *a: raw)
    monkeypatch.setattr(
        f,
        "arm_rows",
        lambda *a: ({"manifest": "fixed", "winners_sha256": "fixed"}, rows),
    )
    monkeypatch.setattr(
        f, "calibration", lambda *a: [{"task_id": str(i)} for i in range(200)]
    )
    result = f.build(tmp_path, spec, crossed_cfg)
    for name, item in result["monitors"].items():
        assert len(item["requests"]) == 1
        assert all(len(b["jobs"]) == 202 for b in item["batches"].values())
        monitor = f.crossed.monitor(name, crossed_cfg)
        body_prompt = f.score_prompt(prompt) if monitor.mode == "score" else prompt
        assert item["worst_case_usd"] == f.reservation_monitor(monitor).worst_case_usd(
            len(body_prompt.encode()) + 2048
        )


def test_kimi_reservation_uses_allowed_routing_price_ceilings():
    _, cfg = f.settings()
    m = f.crossed.monitor("kimi-k3", cfg)
    bounded = f.reservation_monitor(m)
    assert bounded.usd_per_m_prompt == 1
    assert bounded.usd_per_m_completion == 16
    assert bounded.worst_case_usd(10000) == (10000 + 64 * 16) / 1e6
    assert m.usd_per_m_prompt == 0.65  # request/nominal identity is unchanged


@pytest.mark.parametrize("target", sorted(f.TARGETS))
@pytest.mark.parametrize("field", ["monitors", "calibration_artifacts"])
def test_settings_reject_each_target_or_calibration_omission(
    target, field, tmp_path, monkeypatch
):
    import json

    spec, _ = f.settings()
    del spec[field][target]
    (tmp_path / "frontier_config.json").write_text(json.dumps(spec))
    monkeypatch.setattr(f, "HERE", tmp_path)
    with pytest.raises(ValueError, match="all three"):
        f.settings()


def test_parse_failure_with_returned_drift_stops_before_second_dispatch(
    tmp_path, monkeypatch
):
    p = plan()
    p["monitors"]["kimi-k3"]["requests"] = [
        {"id": "first", "prompt": "one", "metadata": {}},
        {"id": "second", "prompt": "two", "metadata": {}},
    ]
    p["monitors"]["kimi-k3"]["identity"]["settings"]["api"] = "openrouter"
    _, cfg = f.settings()
    calls = []

    class MockCampaign(Ledger):
        def request_id(self, *a):
            return str(len(a))

        def reserve(self, *a, **kw):
            return {"id": "test", "charge": 0.5}

        def settle(self, *a):
            pass

    def drift(*a):
        calls.append(1)
        return {
            "id": "first",
            "status": "infrastructure_failure",
            "model": "wrong-model",
            "provider": "makora",
        }

    monkeypatch.setattr(f, "Campaign", lambda *a: MockCampaign())
    monkeypatch.setattr(f, "committed", lambda *a: None)
    monkeypatch.setattr(f, "load_key", lambda *a: "mock-key")
    monkeypatch.setattr(f.crossed, "request_record", drift)
    with pytest.raises(ValueError, match="drift"):
        f.run(p, tmp_path, config(), cfg)
    assert len(calls) == 1


@pytest.mark.parametrize("choices", [[], None])
def test_real_parser_preserves_drift_before_choices(tmp_path, monkeypatch, choices):
    import requests

    p = plan()
    p["monitors"]["kimi-k3"]["requests"] = [
        {"id": "first", "prompt": "one", "metadata": {}},
        {"id": "second", "prompt": "two", "metadata": {}},
    ]
    _, cfg = f.settings()
    calls = []

    class MockCampaign(Ledger):
        def request_id(self, *a):
            return "test"

        def reserve(self, *a, **kw):
            return {"id": "test", "charge": 0.5}

        def settle(self, *a):
            pass

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            data = {"model": "wrong-model", "provider": "makora"}
            if choices is not None:
                data["choices"] = choices
            return data

    def post(*a, **kw):
        calls.append(1)
        return Response()

    monkeypatch.setattr(f, "Campaign", lambda *a: MockCampaign())
    monkeypatch.setattr(f, "committed", lambda *a: None)
    monkeypatch.setattr(f, "load_key", lambda *a: "mock-key")
    monkeypatch.setattr(requests, "post", post)
    with pytest.raises(ValueError, match="drift"):
        f.run(p, tmp_path, config(), cfg)
    assert len(calls) == 1
    saved = f.read(tmp_path / "kimi-k3.jsonl")
    assert saved[0]["model"] == "wrong-model"
    assert saved[0]["provider"] == "makora"


def test_string_batch_errors_are_terminal_without_attribute_error():
    f.check_terminal(
        {"status": "refusal", "error": "refused"},
        {"settings": {"api": "anthropic-batch"}},
    )


def test_prior_target_authentication_failure_blocks_later_target(tmp_path, monkeypatch):
    p, cfg, cc = setup_run(tmp_path, monkeypatch)
    f.append(
        tmp_path / "kimi-k3.jsonl",
        {"id": "0", "status": "infrastructure_failure", "error": {"http_status": 401}},
    )
    monkeypatch.setattr(f, "load_key", lambda *a: pytest.fail("key loaded"))
    with pytest.raises(SystemExit, match="authentication"):
        f.run(p, tmp_path, cfg, cc, "gpt-6.1-sol")


@pytest.mark.parametrize("target", ["gpt-6.1-sol", "claude-sonnet-5.5"])
def test_predecessor_must_finish_before_later_target(tmp_path, monkeypatch, target):
    p, cfg, cc = setup_run(tmp_path, monkeypatch)
    monkeypatch.setattr(f, "load_key", lambda *a: pytest.fail("key loaded"))
    with pytest.raises(SystemExit, match="finish kimi-k3"):
        f.run(p, tmp_path, cfg, cc, target)


def test_prior_target_drift_blocks_later_target(tmp_path, monkeypatch):
    p, cfg, cc = setup_run(tmp_path, monkeypatch)
    f.append(
        tmp_path / "kimi-k3.jsonl",
        {"id": "0", "status": "ok", "model": "wrong-model", "provider": "makora"},
    )
    monkeypatch.setattr(f, "load_key", lambda *a: pytest.fail("key loaded"))
    with pytest.raises(ValueError, match="drift"):
        f.run(p, tmp_path, cfg, cc, "gpt-6.1-sol")
