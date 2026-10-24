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
                "identity": {"settings": s},
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


def test_combined_openrouter_cap_and_campaign_unknown_reservations():
    p = plan()
    assert f.preflight(p, Ledger())["status"] == "ready"
    p["monitors"]["gpt-6.1-sol"]["worst_case_usd"] = 1.6
    gate = f.preflight(p, Ledger())
    assert gate["status"] == "blocked_budget"
    assert "openrouter A4 cap exceeded" in gate["reasons"]
    p = plan()
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
    assert (
        "openrouter shared campaign cap exceeded" in f.preflight(p, ledger)["reasons"]
    )


def test_over_cap_run_stops_before_keys_reservations_or_transport(
    tmp_path, monkeypatch
):
    p = plan()
    p["monitors"]["claude-sonnet-5.5"]["worst_case_usd"] = 1.51
    monkeypatch.setattr(f, "Campaign", lambda *a: Ledger())
    monkeypatch.setattr(f, "load_key", lambda *a: pytest.fail("key loaded"))
    monkeypatch.setattr(f, "committed", lambda *a: pytest.fail("dispatch gate crossed"))
    with pytest.raises(SystemExit, match="nothing sent"):
        f.run(p, tmp_path, config(), {})
    assert not (tmp_path / "dispatch_intent.json").exists()


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
