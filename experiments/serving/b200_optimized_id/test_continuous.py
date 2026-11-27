"""Continuous refill, concurrency bounds and prompt-aware crash recovery."""

import asyncio
import hashlib
import json

import httpx
import pytest

from gleipnir.serving.admission import continuous_score_trial, recovery_records
from gleipnir.serving.monitor_score import score_payload

SETTINGS = {"port": 8010, "timeout_seconds": 5}


def rows():
    return [
        {
            "id": x,
            "prompt": x,
            "prompt_sha256": hashlib.sha256(x.encode()).hexdigest(),
            "prompt_tokens": 2,
        }
        for x in "abcd"
    ]


def test_refills_before_slowest_request_and_preserves_input_order(tmp_path):
    async def scenario():
        admitted = []
        release = asyncio.Event()
        active, peak = 0, 0

        async def handler(request):
            nonlocal active, peak
            name = json.loads(request.content)["prompt"]
            admitted.append(name)
            active += 1
            peak = max(peak, active)
            if name == "a":
                await asyncio.wait_for(release.wait(), 2)
            if name == "c":
                release.set()
            await asyncio.sleep(0)
            active -= 1
            return httpx.Response(200, json=score_payload([0.0, 1.0], 2))

        values, seconds, receipt = await continuous_score_trial(
            rows(),
            2,
            SETTINGS,
            checkpoint=tmp_path / "scores.jsonl",
            contract={},
            transport=httpx.MockTransport(handler),
        )
        assert admitted[:3] == ["a", "b", "c"]
        assert peak == 2
        assert [r["id"] for r in values] == list("abcd")
        assert receipt["new_rows"] == 4 and seconds > 0
        assert all(
            r["completion_offset_seconds"] >= r["start_offset_seconds"] for r in values
        )
        checkpoint = [
            json.loads(x) for x in (tmp_path / "scores.jsonl").read_text().splitlines()
        ]
        assert checkpoint[1]["value"]["id"] == "b"

    asyncio.run(scenario())


def test_partial_failure_drains_writer_and_resume_skips_saved_rows(tmp_path):
    async def scenario():
        checkpoint = tmp_path / "scores.jsonl"

        async def failing(request):
            name = json.loads(request.content)["prompt"]
            return httpx.Response(
                503 if name == "b" else 200, json=score_payload([0.0, 1.0], 2)
            )

        with pytest.raises(httpx.HTTPStatusError):
            await continuous_score_trial(
                rows(),
                1,
                SETTINGS,
                checkpoint=checkpoint,
                contract={"adapter": "frozen"},
                transport=httpx.MockTransport(failing),
            )
        persisted = checkpoint.read_text().splitlines()
        assert len(persisted) == 2
        # A killed append can leave a final partial line; validate the prefix and
        # discard only that tail before continuing the identical contract.
        with checkpoint.open("ab") as handle:
            handle.write(b'{"index":1,')
        seen = []

        async def successful(request):
            seen.append(json.loads(request.content)["prompt"])
            return httpx.Response(200, json=score_payload([0.0, 1.0], 2))

        values, _, receipt = await continuous_score_trial(
            rows(),
            1,
            SETTINGS,
            checkpoint=checkpoint,
            contract={"adapter": "frozen"},
            transport=httpx.MockTransport(successful),
        )
        assert seen == list("bcd")
        assert [r["id"] for r in values] == list("abcd")
        assert receipt["resumed_rows"] == 1
        assert receipt["new_rows"] == 3
        saved_before = checkpoint.read_bytes()
        with pytest.raises(ValueError, match="contract changed"):
            await continuous_score_trial(
                rows(),
                1,
                SETTINGS,
                checkpoint=checkpoint,
                contract={"adapter": "other"},
                transport=httpx.MockTransport(successful),
            )
        assert checkpoint.read_bytes() == saved_before

    asyncio.run(scenario())


def test_recovery_rejects_changed_scores_and_duplicate_indices(tmp_path):
    path = tmp_path / "scores.jsonl"
    header = {"contract": "frozen"}
    value = rows()[0] | score_payload([0.0, 1.0], 2)
    record = {"index": 0, "value": value}
    path.write_text(json.dumps(header) + "\n" + json.dumps(record) + "\n")
    assert len(recovery_records(path, header, rows())[0]) == 1
    value["score"] = 0.1
    path.write_text(json.dumps(header) + "\n" + json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="inconsistent"):
        recovery_records(path, header, rows())
    value["score"] = score_payload([0.0, 1.0], 2)["score"]
    path.write_text(json.dumps(header) + "\n" + (json.dumps(record) + "\n") * 2)
    with pytest.raises(ValueError, match="duplicate"):
        recovery_records(path, header, rows())


def test_writer_failure_cancels_requests_without_hanging(tmp_path, monkeypatch):
    from gleipnir.serving import admission

    fsync_calls = 0

    def failing_sync(_):
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls > 1:
            raise OSError("checkpoint disk unavailable")

    monkeypatch.setattr(admission.os, "fsync", failing_sync)

    async def scenario():
        async def handler(request):
            await asyncio.sleep(0.01)
            return httpx.Response(200, json=score_payload([0.0, 1.0], 2))

        with pytest.raises(OSError, match="disk unavailable"):
            await asyncio.wait_for(
                continuous_score_trial(
                    rows(),
                    2,
                    SETTINGS,
                    checkpoint=tmp_path / "scores.jsonl",
                    contract={},
                    transport=httpx.MockTransport(handler),
                ),
                2,
            )

    asyncio.run(scenario())
