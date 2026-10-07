"""Live tokenization must run for every request and preserve serving options."""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest

from experiments.b200_inference_benchmark.gigatoken import LiveTokenizingClient


def test_live_encoder_runs_each_time_and_preserves_request_options():
    class Encoder:
        calls = 0

        def encode(self, prompt, **kwargs):
            assert kwargs == dict(
                add_special_tokens=False, truncation=True, max_length=32768
            )
            self.calls += 1
            return [15, 16]

    class Client:
        async def post(self, path, *, json):
            assert path == "/v1/completions"
            assert json == {"prompt": [15, 16], "max_tokens": 1}
            return "response"

    async def run():
        original = {"prompt": "text", "max_tokens": 1}
        encoder = Encoder()
        with ThreadPoolExecutor(max_workers=1) as executor:
            client = LiveTokenizingClient(
                Client(), encoder, executor, {"text": [15, 16]}
            )
            for _ in range(2):
                assert await client.post("/v1/completions", json=original) == "response"
            assert encoder.calls == 2
            assert len(client.encodes) == 2
            assert original["prompt"] == "text"
            assert all(
                e["encode_and_queue_seconds"] >= e["encode_seconds"] > 0
                for e in client.encodes
            )

    asyncio.run(run())


def test_live_encoder_rejects_token_drift_before_http_submission():
    class Encoder:
        def encode(self, prompt, **kwargs):
            return [99]

    async def run():
        with ThreadPoolExecutor(max_workers=1) as executor:
            client = LiveTokenizingClient(None, Encoder(), executor, {"text": [15]})
            with pytest.raises(ValueError, match="live Gigatoken ID drift"):
                await client.post("/v1/completions", json={"prompt": "text"})

    asyncio.run(run())
