"""Validate request-scoped Lens interventions and measure their serving overhead."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

from experiments.b200_vllm_lens.start import OVERLAY, ROOT, SERVING, start, write
from gleipnir.serving.benchmark import (
    measurement_summary,
    paired_score_summary,
    ranking_comparison,
)
from gleipnir.serving.monitor_score import validate_score_response


async def trial(
    rows: list[dict], concurrency: int, *, capture: bool = False
) -> tuple[list, float]:
    import torch
    from vllm_lens._helpers._serialize import deserialize_tensor

    limit = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(
        base_url="http://127.0.0.1:8010",
        trust_env=False,
        timeout=300,
        limits=httpx.Limits(max_connections=256, max_keepalive_connections=128),
    ) as client:

        async def one(row):
            async with limit:
                payload = {"model": "monitor", "prompt": row["prompt"]}
                endpoint = "/v1/monitor/score"
                if capture:
                    payload.update(capture_layers=[31], capture_positions="last")
                    endpoint = "/v1/monitor/lens"
                before = time.perf_counter()
                response = await client.post(endpoint, json=payload)
                response.raise_for_status()
                elapsed = time.perf_counter() - before
                value = response.json()
                extra = {}
                if capture:
                    h = deserialize_tensor(value["activations"]["residual_stream"])
                    if (
                        h.shape != (1, 1, 2560)
                        or h.dtype != torch.bfloat16
                        or not bool(torch.isfinite(h).all())
                        or value["activation_positions"] != [row["prompt_tokens"] - 1]
                    ):
                        raise ValueError(
                            "capture shape, dtype, finite or token identity failed"
                        )
                    extra = {
                        "capture_shape": list(h.shape),
                        "capture_chunks": value["capture_chunks"],
                    }
                return {
                    "id": row["id"],
                    "prompt_sha256": row["prompt_sha256"],
                    "prompt_tokens": row["prompt_tokens"],
                    "latency_seconds": elapsed,
                    **validate_score_response(value, row["prompt_tokens"]),
                    **extra,
                }

        before = time.perf_counter()
        values = await asyncio.gather(*(one(row) for row in rows))
        return values, time.perf_counter() - before


async def measure(
    out: Path, rows: list[dict], c: int, *, capture: bool = False
) -> tuple[list, list]:
    directory = out / ("capture" if capture else "plain")
    warm, _ = await trial(rows, c, capture=capture)
    write(directory / f"c{c}_warmup.json", warm)
    values, summaries = [], []
    for repeat in range(3):
        run, seconds = await trial(rows, c, capture=capture)
        write(directory / f"c{c}_repeat{repeat}.json", run)
        summary = {
            "concurrency": c,
            "repeat": repeat,
            **measurement_summary(run, seconds),
        }
        values.append(run)
        summaries.append(summary)
        write(directory / f"c{c}_timing.json", summaries)
        print(
            "lens_benchmark_pass",
            directory.name,
            c,
            repeat,
            summary["prompt_tokens_per_second"],
            flush=True,
        )
    return values, summaries


async def run(name: str, compiled_baseline: Path | None = None) -> None:
    from experiments.b200_long_context.run import gpu
    from experiments.b200_vllm031.run import archive_audits
    from experiments.b200_vllm_lens.smoke import smoke
    from gleipnir.serving.reference import selected_serving_default

    out = ROOT / "results/b200_vllm_lens" / name
    out.mkdir(parents=True, exist_ok=False)
    report = {"status": "starting", "promoted": False}
    write(out / "summary.json", report)
    settings = json.loads(Path(__file__).with_name("config.json").read_text())
    populations = {}
    for name_, contract in settings["workloads"].items():
        path = ROOT / contract["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != contract["sha256"]:
            raise ValueError("frozen inference workload changed")
        rows = json.loads(path.read_text())
        if (
            len(rows) != contract["rows"]
            or sum(r["prompt_tokens"] for r in rows) != contract["prompt_tokens"]
        ):
            raise ValueError("frozen inference workload population changed")
        populations[name_] = rows
    full, quick = populations["full"], populations["quick"]
    write(out / "settings.json", settings)
    source_paths = list(Path(__file__).parent.glob("*.py")) + [
        Path(__file__).with_name(n)
        for n in ("README.md", "config.json", "requirements.txt")
    ]
    source_paths += list((ROOT / "src/gleipnir/serving").glob("lens*.py"))
    for path in source_paths:
        target = out / "executed_sources" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    selection, expected_command = selected_serving_default(ROOT)
    if settings["gpu_uuid"] not in gpu()["gpu"]:
        raise ValueError("authorized B200 identity changed")
    if compiled_baseline is None:
        active = json.loads((SERVING / "server.json").read_text())
        if (
            active.get("lens_research")
            or active["status"] != "ready"
            or active.get("serving_default") != selection["name"]
            or active["command"] != expected_command
        ):
            raise ValueError(
                "Lens requires the selected compiled reference before replacement"
            )
        write(out / "compiled_server.json", active)
        compiled_quick, _ = await measure(out / "compiled", quick, 1)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.b200_vllm031.stop",
                "--archive-name",
                name + "_compiled_parent",
            ],
            cwd=ROOT,
            check=True,
        )
    else:
        active = json.loads((compiled_baseline / "compiled_server.json").read_text())
        if active["command"] != expected_command:
            raise ValueError("reused compiled baseline recipe changed")
        compiled_quick = [
            json.loads(
                (compiled_baseline / f"compiled/plain/c1_repeat{i}.json").read_text()
            )
            for i in range(3)
        ]
        write(out / "compiled_server.json", active)
        write(out / "reused_compiled_baseline.json", {"path": str(compiled_baseline)})
    compiled_full = [
        json.loads(
            (
                ROOT / f"results/b200_score_scaling/nc2_fp8_04/c128_repeat{i}.json"
            ).read_text()
        )
        for i in range(3)
    ]
    paired_score_summary(compiled_full, compiled_full)
    try:
        await start(out)
        report["status"] = "smoke"
        write(out / "summary.json", report)
        write(out / "smoke.json", await smoke(out))
        report["status"] = "measuring"
        write(out / "summary.json", report)
        observations = {}
        for capture in (False, True):
            for c, rows, compiled in (
                (1, quick, compiled_quick),
                (128, full, compiled_full),
            ):
                values, timings = await measure(out, rows, c, capture=capture)
                observations[c, capture] = values
                write(
                    out
                    / ("capture" if capture else "plain")
                    / f"c{c}_compiled_comparison.json",
                    {
                        "scores": paired_score_summary(compiled, values),
                        "ranking": ranking_comparison(rows, compiled, values),
                        "timings": timings,
                    },
                )
        for c, rows in ((1, quick), (128, full)):
            write(
                out / "capture" / f"c{c}_eager_comparison.json",
                {
                    "scores": paired_score_summary(
                        observations[c, False], observations[c, True]
                    ),
                    "ranking": ranking_comparison(
                        rows, observations[c, False], observations[c, True]
                    ),
                },
            )
        archive_audits(out)
        report.update(status="complete", server_retained_warm=True)
        write(out / "summary.json", report)
        write(out / "closure.json", gpu())
        print("monitor_lens_complete", name, flush=True)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        write(out / "summary.json", report)
        if (SERVING / "server.json").exists():
            current = json.loads((SERVING / "server.json").read_text())
            if current.get("lens_research") and current["log"].endswith(
                f"/{name}_server.log"
            ):
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "experiments.b200_vllm031.stop",
                        "--archive-name",
                        name + "_failed",
                    ],
                    cwd=ROOT,
                    check=True,
                )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--compiled-baseline", type=Path)
    args = parser.parse_args()
    if not args.name or Path(args.name).name != args.name or args.name in {".", ".."}:
        raise ValueError("run name must be a stem")
    sys.path.insert(0, str(OVERLAY))
    os.environ["VLLM_LENS_DISABLE"] = "1"
    asyncio.run(run(args.name, args.compiled_baseline))


if __name__ == "__main__":
    main()
