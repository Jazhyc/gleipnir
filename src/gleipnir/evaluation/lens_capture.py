"""Resumable two-surface Lens scoring and lossless final-token residual capture."""

from __future__ import annotations

import asyncio
import json
import math
import time
from pathlib import Path
from typing import Literal

import httpx
import numpy as np

from gleipnir.data.monitoring import file_hash, write_json, write_rows
from gleipnir.serving.monitor_score import validate_score_response


def binary_probability(logits: list[float]) -> float:
    """Return conditional probability of the second literal token, preserving ties."""
    if len(logits) != 2 or not all(math.isfinite(v) for v in logits):
        raise ValueError("invalid binary readout")
    margin = logits[1] - logits[0]
    return (
        1 / (1 + math.exp(-margin))
        if margin >= 0
        else math.exp(margin) / (1 + math.exp(margin))
    )


async def capture_rows(
    rows: list[dict],
    out: Path,
    population: str,
    *,
    surface: Literal["01", "AB"],
    unit: np.ndarray,
    intervention: dict | None = None,
    concurrency: int = 32,
    batch_rows: int = 128,
    timeout_seconds: float = 300,
    max_projected_component_fraction: float | None = None,
) -> list[dict]:
    """Capture audited Qwen4B layers 20/31, retaining both head readouts explicitly."""
    from vllm_lens._helpers._serialize import deserialize_tensor

    if (
        surface not in ["01", "AB"]
        or unit.shape != (2560,)
        or not np.isfinite(unit).all()
    ):
        raise ValueError("invalid surface/direction")
    folder = out / "batches" / population
    folder.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    all_values = []
    async with httpx.AsyncClient(
        base_url="http://127.0.0.1:8010",
        trust_env=False,
        timeout=timeout_seconds,
        limits=httpx.Limits(max_connections=concurrency),
    ) as client:

        async def one(row):
            payload = {
                "model": "monitor",
                "prompt": row["prompt"],
                "capture_layers": [20, 31],
                "capture_positions": "last",
                "full_readout": True,
            }
            if intervention:
                payload.update(intervention)
            started = time.perf_counter()
            async with sem:
                response = await client.post("/v1/monitor/lens", json=payload)
            response.raise_for_status()
            body = response.json()
            native = validate_score_response(body, row["prompt_tokens"])
            if body["activation_layers"] != [20, 31] or body[
                "activation_positions"
            ] != [row["prompt_tokens"] - 1]:
                raise ValueError("capture identity changed")
            residual = (
                deserialize_tensor(body["activations"]["residual_stream"])
                .float()
                .numpy()[:, 0]
            )
            if residual.shape != (2, 2560) or not np.isfinite(residual).all():
                raise ValueError("invalid residual capture")
            judge = body["judge_readout"]
            if judge["token_ids"] != [32, 33]:
                raise ValueError("A/B token identity changed")
            score_ab = binary_probability(judge["logits"])
            if (
                max(
                    abs(a - b)
                    for a, b in zip(body["logits"], body["readout_logits"], strict=True)
                )
                > 0.25
            ):
                raise ValueError("native/full 0/1 readout mismatch")
            if not all(
                math.isfinite(v) and 0 <= v <= 1
                for v in [body["p01"], judge["probability_mass"]]
            ):
                raise ValueError("invalid answer mass")
            logits = body["logits"] if surface == "01" else judge["logits"]
            value = {
                k: row[k] for k in ["id", "prompt_sha256", "prompt_tokens"]
            } | row.get("metadata", {})
            value.update(
                surface=surface,
                score=native["score"] if surface == "01" else score_ab,
                logits=logits,
                margin=logits[1] - logits[0],
                logits_01=body["logits"],
                score_01=native["score"],
                logits_01_full=body["readout_logits"],
                score_01_full=binary_probability(body["readout_logits"]),
                logits_ab=judge["logits"],
                score_ab=score_ab,
                p01=body["p01"],
                pab=judge["probability_mass"],
                latency_seconds=time.perf_counter() - started,
            )
            for i, layer in enumerate([20, 31]):
                z = float(residual[i] @ unit)
                norm = float(np.linalg.norm(residual[i]))
                if norm <= 0 or not math.isfinite(norm) or not math.isfinite(z):
                    raise ValueError("invalid direction readout")
                if (
                    max_projected_component_fraction is not None
                    and abs(z) / norm > max_projected_component_fraction
                ):
                    raise ValueError("projection component too large")
                value[f"z{layer}"] = z
                value[f"norm{layer}"] = norm
            return value, residual

        for offset in range(0, len(rows), batch_rows):
            batch = rows[offset : offset + batch_rows]
            path = folder / f"{offset:06d}.json"
            tensor = path.with_suffix(".npz")
            if path.exists():
                saved = json.loads(path.read_text())
                values = saved["values"]
                if file_hash(tensor) != saved["tensor_sha256"] or [
                    (v["id"], v["prompt_sha256"], v["prompt_tokens"]) for v in values
                ] != [(r["id"], r["prompt_sha256"], r["prompt_tokens"]) for r in batch]:
                    raise ValueError("resume identity drift")
            else:
                started = time.perf_counter()
                results = await asyncio.gather(*(one(r) for r in batch))
                values = [v for v, a in results]
                np.savez(tensor, residual=np.stack([a for v, a in results]))
                write_json(
                    path,
                    {
                        "values": values,
                        "seconds": time.perf_counter() - started,
                        "tokens": sum(r["prompt_tokens"] for r in batch),
                        "tensor_sha256": file_hash(tensor),
                    },
                )
            all_values.extend(values)
            write_json(
                out / "status.json",
                {
                    "stage": "scoring",
                    "population": population,
                    "rows": len(all_values),
                    "total": len(rows),
                },
            )
            print(
                "lens_capture_progress",
                population,
                len(all_values),
                len(rows),
                flush=True,
            )
    write_rows(out / (population + ".jsonl"), all_values)
    return all_values
