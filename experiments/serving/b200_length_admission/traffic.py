"""Replay a deterministic mixed-length arrival trace against a ready scorer."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import time
from pathlib import Path

import httpx

from experiments.b200_inference_benchmark.run import DATA, ROOT, sha, write
from gleipnir.serving.benchmark import (
    measurement_summary,
    percentile,
    ranking_comparison,
)
from gleipnir.serving.monitor_score import ENDPOINT, validate_score_response


def arrival_offsets(count: int, rate: float, seed: int) -> list[float]:
    """Freeze Poisson arrival times; first request starts at zero."""
    if count <= 0 or not math.isfinite(rate) or rate <= 0:
        raise ValueError("arrival count/rate must be positive and finite")
    rng = random.Random(seed)
    offsets = [0.0]
    for _ in range(count - 1):
        offsets.append(offsets[-1] + rng.expovariate(rate))
    return offsets


async def replay(
    rows: list[dict], offsets: list[float], port: int
) -> tuple[list[dict], float]:
    """Avoid a client semaphore; include dispatch lag in arrival-to-response time."""
    if len(rows) != len(offsets):
        raise ValueError("arrival trace length mismatch")
    async with httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{port}",
        trust_env=False,
        timeout=180,
        limits=httpx.Limits(
            max_connections=len(rows), max_keepalive_connections=len(rows)
        ),
    ) as client:
        start = time.perf_counter()

        async def one(row: dict, offset: float) -> dict:
            deadline = start + offset
            await asyncio.sleep(max(0, deadline - time.perf_counter()))
            sent = time.perf_counter()
            response = await client.post(
                ENDPOINT, json={"model": "monitor", "prompt": row["prompt"]}
            )
            response.raise_for_status()
            done = time.perf_counter()
            return {
                "id": row["id"],
                "prompt_sha256": row["prompt_sha256"],
                "prompt_tokens": row["prompt_tokens"],
                "arrival_offset_seconds": offset,
                "dispatch_lag_seconds": sent - deadline,
                "http_latency_seconds": done - sent,
                "latency_seconds": done - deadline,
                **validate_score_response(response.json(), row["prompt_tokens"]),
            }

        values = await asyncio.gather(
            *(one(r, t) for r, t in zip(rows, offsets, strict=True))
        )
        return values, time.perf_counter() - start


async def run(args: argparse.Namespace) -> None:
    if args.name in {"", ".", ".."} or Path(args.name).name != args.name:
        raise ValueError("run name must be a directory stem")
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    manifest = json.loads((DATA / "manifest.json").read_text())
    if sha(DATA / "full.json") != manifest["files"]["full"]:
        raise ValueError("frozen full workload drift")
    rows = json.loads((DATA / "full.json").read_text())
    settings = json.loads((Path(__file__).parent / "config.json").read_text())
    rates = settings["mixed_traffic"]["requests_per_second"]
    seed = settings["mixed_traffic"]["seed"]
    contract = {
        "manifest_sha256": sha(DATA / "manifest.json"),
        "seed": seed,
        "rates": rates,
        "repeats": args.repeats,
        "arrivals": {str(r): arrival_offsets(len(rows), r, seed) for r in rates},
        "order": [{"id": r["id"], "prompt_sha256": r["prompt_sha256"]} for r in rows],
    }
    contract_hash = hashlib.sha256(
        json.dumps(contract, sort_keys=True).encode()
    ).hexdigest()
    reference = None
    if args.reference:
        reference = json.loads((args.reference / "summary.json").read_text())
        if (
            reference.get("status") != "complete"
            or reference["contract_sha256"] != contract_hash
        ):
            raise ValueError("mixed-traffic control has a different arrival contract")
    out = ROOT / "results/b200_length_admission" / args.name
    out.mkdir(parents=True, exist_ok=False)
    write(out / "contract.json", contract)
    write(
        out / "sources.json",
        {p: sha(ROOT / p) for p in settings["additional_sources"]},
    )
    server_path = ROOT / "results/b200_attention_gdn_serving/server.json"
    server = json.loads(server_path.read_text())
    if server.get("status") != "ready" or server.get("endpoint") != ENDPOINT:
        raise ValueError("traffic replay requires a ready monitoring scorer")
    if int(server["command"][server["command"].index("--port") + 1]) != args.port:
        raise ValueError("traffic port differs from the recorded scorer")
    write(out / "server.json", server)
    summary = {"contract_sha256": contract_hash, "trials": [], "promoted": False}
    for rate in rates:
        offsets = contract["arrivals"][str(rate)]
        # Warm the same prompt mix outside timed arrivals.
        await replay(rows, [0.0] * len(rows), args.port)
        candidates = []
        for repeat in range(args.repeats):
            values, seconds = await replay(rows, offsets, args.port)
            candidates.append(values)
            write(out / f"r{rate}_repeat{repeat}.json", values)
            metrics = measurement_summary(values, seconds)
            metrics["max_dispatch_lag_seconds"] = max(
                v["dispatch_lag_seconds"] for v in values
            )
            metrics["p95_dispatch_lag_seconds"] = percentile(
                [v["dispatch_lag_seconds"] for v in values], 0.95
            )
            metrics["arrival_timing_valid"] = (
                metrics["p95_dispatch_lag_seconds"] <= 0.005
            )
            summary["trials"].append(
                {"arrival_rate": rate, "repeat": repeat, **metrics}
            )
            write(out / "summary.json", summary)
        if reference is not None:
            controls = [
                json.loads((args.reference / f"r{rate}_repeat{i}.json").read_text())
                for i in range(args.repeats)
            ]
            write(
                out / f"r{rate}_quality.json",
                ranking_comparison(rows, controls, candidates),
            )
    summary["status"] = "complete"
    write(out / "summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except BaseException as error:
        out = ROOT / "results/b200_length_admission" / args.name
        if (
            out.exists()
            and args.name not in {"", ".", ".."}
            and (Path(args.name).name == args.name)
        ):
            write(out / "failure.json", {"error": f"{type(error).__name__}: {error}"})
        raise


if __name__ == "__main__":
    main()
