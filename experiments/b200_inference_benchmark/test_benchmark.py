"""Focused contracts for workload lineage and production scoring measurements."""

import asyncio
import json

import pytest

from gleipnir.inference_benchmark import (
    match_selection,
    measurement_summary,
    quick_workload,
    response_score,
)


def test_exact_selection_rejects_lineage_drift():
    rows = [{"dataset": "a", "index": "1", "label": 0, "trajectory_sha256": "hash"}]
    assert match_selection(rows, rows) == rows
    with pytest.raises(ValueError, match="selection drift"):
        match_selection(rows, [{**rows[0], "trajectory_sha256": "other"}])
    with pytest.raises(ValueError, match="duplicate"):
        match_selection(rows * 2, rows)


def test_quick_workload_is_frozen_and_covers_length_extremes():
    rows = [{"id": str(i), "prompt_tokens": i + 1} for i in range(320)]
    selected = quick_workload(rows, 64, 0)
    assert selected == quick_workload(rows, 64, 0)
    assert len({r["id"] for r in selected}) == 64
    assert min(r["prompt_tokens"] for r in selected) == 1
    assert max(r["prompt_tokens"] for r in selected) == 320
    assert {r["id"] for r in quick_workload(rows, 320, 0)} == {r["id"] for r in rows}


def test_legacy_rows_without_optional_lineage_remain_checksum_bound():
    legacy = {"dataset": "legacy", "index": "1", "label": 0}
    assert match_selection([legacy], [legacy]) == [legacy]
    with pytest.raises(ValueError, match="trajectory_sha256"):
        match_selection([legacy], [{**legacy, "trajectory_sha256": "hash"}])


def response():
    return {
        "usage": {"prompt_tokens": 10, "completion_tokens": 1},
        "choices": [
            {"logprobs": {"top_logprobs": [{"token_id:15": -2.0, "token_id:16": -1.0}]}}
        ],
    }


def test_http_margin_preserves_binary_probability():
    assert response_score(response(), [15, 16], 10)["score"] == pytest.approx(
        0.7310585786
    )


def test_missing_nonfinite_logprobs_or_truncation_fail():
    item = response()
    del item["choices"][0]["logprobs"]["top_logprobs"][0]["token_id:16"]
    with pytest.raises(KeyError):
        response_score(item, [15, 16], 10)
    item = response()
    item["choices"][0]["logprobs"]["top_logprobs"][0]["token_id:16"] = float("-inf")
    with pytest.raises(ValueError, match="nonfinite"):
        response_score(item, [15, 16], 10)
    with pytest.raises(ValueError, match="truncation"):
        response_score(response(), [15, 16], 11)


def test_closed_loop_throughput_and_length_bins():
    rows = [
        {"prompt_tokens": 1000, "latency_seconds": 1.0},
        {"prompt_tokens": 20000, "latency_seconds": 3.0},
    ]
    result = measurement_summary(rows, 4.0)
    assert result["requests_per_second"] == 0.5
    assert result["prompt_tokens_per_second"] == 5250
    assert result["latency"]["p50_seconds"] == 2.0
    assert result["latency_by_prompt_length"]["ge16k"]["p95_seconds"] == 3.0


def test_http_trial_bounds_concurrency_and_preserves_output_order():
    from experiments.b200_inference_benchmark.run import trial

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return response()

    class FakeClient:
        active = 0
        peak = 0

        async def post(self, path, *, json):
            assert path == "/v1/completions"
            assert json["allowed_token_ids"] == [15, 16]
            assert json["max_tokens"] == 1
            assert json["add_special_tokens"] is False
            self.active += 1
            self.peak = max(self.peak, self.active)
            await asyncio.sleep(0)
            self.active -= 1
            return FakeResponse()

    client = FakeClient()
    rows = [
        {"id": str(i), "prompt": "prompt", "prompt_sha256": "hash", "prompt_tokens": 10}
        for i in range(7)
    ]
    results, elapsed = asyncio.run(trial(client, rows, [15, 16], 3))
    assert client.peak == 3
    assert [row["id"] for row in results] == [row["id"] for row in rows]
    assert elapsed > 0


def test_prepared_workload_reuse_checks_rendered_bytes(tmp_path, monkeypatch):
    from experiments.b200_inference_benchmark import run

    monkeypatch.setattr(run, "DATA", tmp_path)
    monkeypatch.setattr(run, "EXPERIMENT", tmp_path)
    (tmp_path / "config.yaml").write_text("frozen")
    (tmp_path / "quick.json").write_text("[]")
    manifest = {
        "config_sha256": run.sha(tmp_path / "config.yaml"),
        "inputs": {"source": "sourcehash"},
        "files": {"quick": run.sha(tmp_path / "quick.json")},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert run.prepared_manifest({"source_sha256": "sourcehash"}) == manifest
    (tmp_path / "quick.json").write_text('["changed"]')
    with pytest.raises(ValueError, match="prepared prompt drift"):
        run.prepared_manifest({"source_sha256": "sourcehash"})
