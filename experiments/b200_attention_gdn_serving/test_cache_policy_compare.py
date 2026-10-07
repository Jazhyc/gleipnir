"""Matched speed summaries must retain score identity and ranking changes."""

import asyncio
import copy

import httpx
import pytest

from experiments.b200_attention_gdn_serving import cache_policy_compare as screen
from experiments.b200_attention_gdn_serving.cache_policy_compare import comparison


def fixture():
    rows = [
        {"id": str(i), "prompt_sha256": str(i), "dataset": "source", "label": y}
        for i, y in enumerate([0, 0, 1, 1])
    ]
    values = [
        {"id": r["id"], "prompt_sha256": r["prompt_sha256"], "score": s, "margin": s}
        for r, s in zip(rows, [0.1, 0.2, 0.8, 0.9], strict=True)
    ]
    report = {
        "values": {"1": [values], "128": [values]},
        "timing": [
            {
                "concurrency": c,
                "prompt_tokens_per_second": rate,
                "latency": {"p50_seconds": 0.1, "p95_seconds": 0.2},
            }
            for c in [1, 128]
            for rate in [100, 200, 900]
        ],
    }
    return rows, report


def test_summary_uses_all_repeat_medians_and_reports_quality_changes():
    rows, baseline = fixture()
    candidate = copy.deepcopy(baseline)
    for timing in candidate["timing"]:
        timing["prompt_tokens_per_second"] *= 2
        timing["latency"]["p50_seconds"] /= 2
    candidate["values"]["128"][0][1]["score"] = 0.85
    report = comparison(rows, baseline, candidate)
    assert report["c128"]["baseline"]["input_tokens_per_second"] == 200
    assert report["c128"]["candidate"]["input_tokens_per_second"] == 400
    assert report["c128"]["throughput_change_percent"] == 100
    assert report["c1"]["median_latency_change_percent"] == -50
    assert report["ranking"]["auroc_delta"]["pooled"] == -0.25
    assert report["c128"]["scores"]["threshold_flips"] == 1


def test_same_lengths_cannot_hide_prompt_identity_drift():
    rows, baseline = fixture()
    candidate = copy.deepcopy(baseline)
    candidate["values"]["1"][0][0]["prompt_sha256"] = "changed"
    with pytest.raises(ValueError, match="identity drift"):
        comparison(rows, baseline, candidate)


def test_failed_connection_pool_is_closed_and_not_reused(monkeypatch):
    clients = []

    class Client:
        def __init__(self, **kwargs):
            self.closed = False
            clients.append(self)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            self.closed = True

    async def trial(client, rows, ids, concurrency):
        if len(clients) == 1:
            raise httpx.ReadError("stale connection")
        return rows, 1.0

    monkeypatch.setattr(screen.httpx, "AsyncClient", Client)
    monkeypatch.setattr(screen, "trial", trial)

    async def run():
        settings = {"port": 8010, "timeout_seconds": 180}
        with pytest.raises(httpx.ReadError):
            await screen.infer([], [15, 16], 128, settings)
        assert await screen.infer([], [15, 16], 128, settings) == ([], 1.0)

    asyncio.run(run())
    assert len(clients) == 2 and all(client.closed for client in clients)
