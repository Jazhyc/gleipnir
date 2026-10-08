"""Ordered, bounded monitor scoring with raw decision logits."""

import asyncio
import time

import httpx

from gleipnir.serving.monitor_score import ENDPOINT, validate_score_response


async def score_batch(rows: list[dict], config: dict) -> tuple[list[dict], float]:
    """Keep raw decision logits and validate complete, finite two-row responses."""
    async with httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{config['port']}",
        trust_env=False,
        timeout=config["timeout_seconds"],
        limits=httpx.Limits(max_connections=config["concurrency"]),
    ) as client:
        semaphore = asyncio.Semaphore(config["concurrency"])

        async def one(row: dict) -> dict:
            before = time.perf_counter()
            async with semaphore:
                response = await client.post(
                    ENDPOINT, json={"model": "monitor", "prompt": row["prompt"]}
                )
            response.raise_for_status()
            payload = response.json()
            value = validate_score_response(payload, row["prompt_tokens"])
            return {
                "id": row["id"],
                "prompt_sha256": row["prompt_sha256"],
                "prompt_tokens": row["prompt_tokens"],
                "logits": payload["logits"],
                "latency_seconds": time.perf_counter() - before,
                **value,
            }

        before = time.perf_counter()
        values = await asyncio.gather(*(one(row) for row in rows))
        return values, time.perf_counter() - before
