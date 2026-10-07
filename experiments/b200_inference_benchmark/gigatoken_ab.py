"""Paired HF/native tokenizer comparisons in one persistent API/GPU worker."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import signal
import statistics
import time
from pathlib import Path

from experiments.b200_inference_benchmark.run import DATA, ROOT, sha, trial, write
from gleipnir.inference_benchmark import (
    measurement_summary,
    paired_score_summary,
    ranking_comparison,
)


class ModeController:
    """Signal only the identity-verified API and wait for an atomic mode receipt."""

    def __init__(self, server: dict, engine_pid: int) -> None:
        self.server = server
        self.engine_pid = engine_pid
        self.receipt = Path(server["frontend"]["receipt_path"])

    def read(self) -> dict:
        value = json.loads(self.receipt.read_text())
        if value["pid"] != self.server["pid"] or "ab_control" not in value:
            raise ValueError("frontend A/B identity or control drift")
        root = ROOT / "results/b200_attention_gdn_serving"
        if json.loads((root / "server.json").read_text())["pid"] != value["pid"]:
            raise ValueError("API process changed during comparison")
        if (
            json.loads((root / "loaded_precision.json").read_text())["worker_pid"]
            != self.engine_pid
        ):
            raise ValueError("GPU engine changed during comparison")
        os.kill(self.engine_pid, 0)
        return value["ab_control"]

    async def toggle(self) -> dict:
        state = self.read()
        actual = [
            v.decode()
            for v in Path(f"/proc/{self.server['pid']}/cmdline")
            .read_bytes()
            .split(b"\0")
            if v
        ]
        if actual != self.server["command"]:
            raise ValueError("API command identity changed; refuse signal")
        os.kill(self.server["pid"], signal.SIGUSR1)
        deadline = time.perf_counter() + 10
        while time.perf_counter() < deadline:
            current = self.read()
            if current["generation"] == state["generation"] + 1:
                if current["mode"] == state["mode"] or current["active_encodes"]:
                    raise ValueError("invalid frontend switch acknowledgement")
                return current
            await asyncio.sleep(0.01)
        raise TimeoutError("frontend switch was not acknowledged")

    async def select(self, mode: str, *, refresh: bool = False) -> dict:
        if mode not in {"hf", "native"}:
            raise ValueError("invalid comparison mode")
        if self.read()["mode"] != mode:
            return await self.toggle()
        if refresh:
            await self.toggle()
            return await self.toggle()
        return self.read()


def paired_ratios(hf: list[float], native: list[float]) -> dict:
    ratios = [b / a for a, b in zip(hf, native, strict=True)]
    rng = random.Random(0)
    bootstrap = sorted(
        statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(10000)
    )
    return {
        "pairs": len(ratios),
        "ratios": ratios,
        "median": statistics.median(ratios),
        "minimum": min(ratios),
        "maximum": max(ratios),
        "bootstrap_95_interval": [bootstrap[249], bootstrap[9749]],
        "interval_scope": "small repeated systems workload; not production traffic",
    }


async def measure(name: str) -> None:
    import httpx

    out = ROOT / "results/b200_inference_benchmark" / name
    out.mkdir(exist_ok=False)
    (out / "executed_client.py").write_bytes(Path(__file__).read_bytes())
    server_root = ROOT / "results/b200_attention_gdn_serving"
    server = json.loads((server_root / "server.json").read_text())
    engine_pid = json.loads((server_root / "loaded_precision.json").read_text())[
        "worker_pid"
    ]
    controller = ModeController(server, engine_pid)
    manifest = json.loads((DATA / "manifest.json").read_text())
    full, quick = (
        json.loads((DATA / f"{name}.json").read_text()) for name in ["full", "quick"]
    )
    for key in ["full", "quick"]:
        if sha(DATA / f"{key}.json") != manifest["files"][key]:
            raise ValueError("frozen workload drift")
    expected = json.loads(
        (
            ROOT / "results/b200_inference_benchmark/tokenization01/token_ids.json"
        ).read_text()
    )
    report = {
        "status": "running",
        "server": server,
        "engine_pid": engine_pid,
        "manifest_sha256": sha(DATA / "manifest.json"),
        "baseline_sha256": sha(Path(__file__).parent / "baseline.json"),
        "source_sha256": sha(Path(__file__)),
        "trials": [],
        "same_api_and_gpu_worker": True,
        "caller_encoding": False,
    }

    async def infer(rows, concurrency):
        async with httpx.AsyncClient(
            base_url="http://127.0.0.1:8010",
            timeout=300,
            trust_env=False,
            limits=httpx.Limits(max_connections=256, max_keepalive_connections=128),
        ) as client:
            return await trial(client, rows, manifest["token_ids"], concurrency)

    try:
        for mode in ["hf", "native"]:
            await controller.select(mode)
            async with httpx.AsyncClient(
                base_url="http://127.0.0.1:8010", timeout=300, trust_env=False
            ) as client:
                for row in full:
                    response = await client.post(
                        "/tokenize",
                        json={
                            "model": "monitor",
                            "prompt": row["prompt"],
                            "add_special_tokens": False,
                        },
                    )
                    response.raise_for_status()
                    if response.json()["tokens"] != expected[row["id"]]:
                        raise ValueError(f"{mode} live token-ID mismatch")
            print("mode_parity_passed", mode, len(full), flush=True)
        report["exact_token_parity_rows_per_mode"] = len(full)
        write(out / "summary.json", report)
        for concurrency, rows, repeats in [(1, quick, 3), (128, full, 6)]:
            results = {"hf": [], "native": []}
            for mode in ["hf", "native"]:
                await controller.select(mode)
                for warmup in range(2):
                    values, seconds = await infer(rows, concurrency)
                    write(
                        out / f"warmup_c{concurrency}_{mode}_{warmup}.json",
                        {"seconds": seconds, "values": values},
                    )
            for pair in range(repeats):
                for mode in ["hf", "native"] if pair % 2 == 0 else ["native", "hf"]:
                    before = await controller.select(mode, refresh=True)
                    values, seconds = await infer(rows, concurrency)
                    after = await controller.select(mode, refresh=True)
                    other = "native" if mode == "hf" else "hf"
                    if (
                        after["calls"][mode] - before["calls"][mode] != len(rows)
                        or after["calls"][other] != before["calls"][other]
                    ):
                        raise ValueError("backend callback count drift")
                    results[mode].append(values)
                    write(out / f"c{concurrency}_{mode}_pair{pair}.json", values)
                    item = {
                        "concurrency": concurrency,
                        "mode": mode,
                        "pair": pair,
                        **measurement_summary(values, seconds),
                        "frontend_before": before,
                        "frontend_after": after,
                        "encoder_seconds": after["encode_seconds"][mode]
                        - before["encode_seconds"][mode],
                    }
                    report["trials"].append(item)
                    write(out / "summary.json", report)
                    print(
                        "paired_pass",
                        concurrency,
                        pair,
                        mode,
                        round(seconds, 3),
                        flush=True,
                    )
            ts = {
                mode: [
                    t
                    for t in report["trials"]
                    if t["concurrency"] == concurrency and t["mode"] == mode
                ]
                for mode in results
            }
            report[f"c{concurrency}_comparison"] = {
                "tokens_s": {
                    mode: statistics.median(
                        t["prompt_tokens_per_second"] for t in values
                    )
                    for mode, values in ts.items()
                },
                "latency_p50": {
                    mode: statistics.median(t["latency"]["p50_seconds"] for t in values)
                    for mode, values in ts.items()
                },
                "latency_p95": {
                    mode: statistics.median(t["latency"]["p95_seconds"] for t in values)
                    for mode, values in ts.items()
                },
                "paired_throughput": paired_ratios(
                    [t["prompt_tokens_per_second"] for t in ts["hf"]],
                    [t["prompt_tokens_per_second"] for t in ts["native"]],
                ),
                "paired_scores": paired_score_summary(results["hf"], results["native"]),
                "ranking": ranking_comparison(rows, results["hf"], results["native"]),
            }
            write(out / "summary.json", report)
        profiles = ROOT / "results/b200_attention_gdn_serving/profiles"
        for index, mode in enumerate(["hf", "native", "hf"]):
            await controller.select(mode)
            existing = {p.name for p in profiles.glob("*.gz")}
            async with httpx.AsyncClient(
                base_url="http://127.0.0.1:8010", timeout=600, trust_env=False
            ) as client:
                (await client.post("/start_profile")).raise_for_status()
                try:
                    values, seconds = await infer(full, 128)
                finally:
                    (await client.post("/stop_profile")).raise_for_status()
            new = [p for p in profiles.glob("*.gz") if p.name not in existing]
            if len(new) != 1:
                raise ValueError("unexpected profile export count")
            target = out / f"profile_{index}_{mode}"
            target.mkdir()
            (target / new[0].name).write_bytes(new[0].read_bytes())
            write(
                target / "pass.json",
                {
                    "mode": mode,
                    "seconds": seconds,
                    "profiling_enabled": True,
                    "timed_benchmark": False,
                },
            )
            write(target / "predictions.json", values)
            print("paired_profile_complete", index, mode, flush=True)
        report.update(status="complete", gpu_recipe_changed=False)
    finally:
        report["restored_mode"] = await controller.select("native", refresh=True)
        write(out / "summary.json", report)
    print("same_worker_ab_complete", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    out = ROOT / "results/b200_inference_benchmark" / args.name
    if Path(args.name).name != args.name or out.exists():
        raise ValueError("use a new output directory stem")
    try:
        asyncio.run(measure(args.name))
    except BaseException as error:
        write(out / "failure.json", {"error": f"{type(error).__name__}: {error}"})
        raise


if __name__ == "__main__":
    main()
