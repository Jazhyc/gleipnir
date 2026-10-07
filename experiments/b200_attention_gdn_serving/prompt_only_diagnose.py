"""Bounded cache-free stall investigation; all timings are diagnostic only."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

import httpx
import yaml

from experiments.b200_attention_gdn_serving.run import OUTPUT, resolve_condition
from experiments.b200_inference_benchmark.run import (
    DATA,
    EXPERIMENT,
    ROOT,
    benchmark,
    prepared_manifest,
    sha,
    trial,
    verify_merged_model,
    write,
)
from gleipnir._compat import canonical_source_reference


async def drive(
    config: dict, manifest: dict, out: Path, repeats: int, mode: str
) -> None:
    rows = json.loads((DATA / "full.json").read_text())
    report = {
        "diagnostic_only": True,
        "timings_excluded": True,
        "status": "running",
        "repeats_completed": 0,
        "manifest_sha256": sha(DATA / "manifest.json"),
    }
    write(out / "diagnostic.json", report)
    async with httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{config['port']}",
        trust_env=False,
        timeout=180,
        limits=httpx.Limits(max_connections=128),
    ) as client:
        response = await client.post(
            "/collective_rpc",
            json={"method": "prompt_only_trace_state", "kwargs": {}, "timeout": 60},
        )
        response.raise_for_status()
        report["trace"] = response.json()["results"][0]
        if report["trace"]["mode"] != mode:
            raise ValueError("requested operator trace mode is not active")
        write(out / "diagnostic.json", report)
        for repeat in range(repeats):
            values, elapsed = await trial(client, rows, manifest["token_ids"], 128)
            write(
                out / f"diagnostic_c128_repeat{repeat}.json",
                {
                    "diagnostic_only": True,
                    "timings_excluded": True,
                    "elapsed_seconds": elapsed,
                    "predictions": values,
                },
            )
            report["repeats_completed"] += 1
            write(out / "diagnostic.json", report)
            print(
                f"diagnostic_repeat_complete repeat={repeat} rows={len(values)}",
                flush=True,
            )
        report["status"] = "complete"
        write(out / "diagnostic.json", report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--condition", type=Path, default=Path(__file__).with_name("prompt_only.json")
    )
    parser.add_argument("--mode", choices=["launch", "sync"], default="launch")
    parser.add_argument("--repeats", type=int, default=6)
    args = parser.parse_args()
    if Path(args.output).name != args.output or not 1 <= args.repeats <= 20:
        parser.error("output must be a directory name; repeats must be in 1..20")
    out = OUTPUT / args.output
    out.mkdir(exist_ok=False)
    base = yaml.safe_load((EXPERIMENT / "config.yaml").read_text())
    manifest = prepared_manifest(base)
    raw = json.loads(args.condition.read_text())
    if raw.get("prompt_only") is not True:
        raise ValueError("diagnostic tracing requires the cache-free worker")
    selection = json.loads((EXPERIMENT / "baseline.json").read_text())
    previous = json.loads((ROOT / selection["results"] / "condition.json").read_text())
    argv = previous["extra_server_args"]
    sources = {
        canonical_source_reference(p)
        for p in json.loads(argv[argv.index("--additional-config") + 1])[
            "gleipnir_frost_fp4"
        ]
    }
    sources.update(
        [
            "src/gleipnir/__init__.py",
            "src/gleipnir/_compat.py",
            "src/gleipnir/serving/prompt_only.py",
            "src/gleipnir/serving/prompt_only_contract.py",
            "experiments/b200_attention_gdn_serving/prompt_only_worker.py",
            "experiments/b200_attention_gdn_serving/prompt_only_canary.py",
        ]
    )
    condition = resolve_condition(raw, {p: sha(ROOT / p) for p in sorted(sources)})
    config = {**base, **raw["serving_config_overrides"], "port": raw["port"]}
    trace_directory = out / "worker_trace"
    os.environ["GLEIPNIR_PROMPT_ONLY_TRACE"] = str(trace_directory)
    os.environ["GLEIPNIR_PROMPT_ONLY_TRACE_MODE"] = args.mode
    write(out / "condition.json", condition)
    write(out / "manifest.json", manifest)
    for source in [
        "src/gleipnir/__init__.py",
        "src/gleipnir/_compat.py",
        *sources,
        str(Path(__file__).relative_to(ROOT)),
        "src/gleipnir/serving/operator_trace.py",
        "experiments/b200_attention_gdn_serving/prompt_only_trace.py",
        "experiments/b200_inference_benchmark/frontend_server.py",
    ]:
        target = out / "executed_sources" / source
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / source).read_bytes())
    write(
        out / "diagnostic.json",
        {
            "diagnostic_only": True,
            "timings_excluded": True,
            "status": "starting",
            "mode": args.mode,
        },
    )
    try:
        merged = Path(raw["merged_model"])
        write(out / "merged_artifact.json", verify_merged_model(base, merged))
        asyncio.run(
            benchmark(
                config, manifest, out, 64, False, merged, condition, startup_only=True
            )
        )
        asyncio.run(drive(config, manifest, out, args.repeats, args.mode))
    except BaseException as error:
        write(
            out / "failure.json",
            {"diagnostic_only": True, "error": f"{type(error).__name__}: {error}"},
        )
        raise


if __name__ == "__main__":
    main()
