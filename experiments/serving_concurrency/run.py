"""Time one serving condition across client concurrency, then retire it."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "results/serving_concurrency"
LOGS = ROOT / "logs/runpod/serving_concurrency"
STEPS = {
    "4b": ("bf16", "quant", "lastquery", "route", "fast"),
    "27b": ("bf16", "quant", "fast"),
}
SWEEP = (128, 64, 32, 16, 8, 4, 2, 1)
ENDS = (128, 1)
LENGTHS = (256, 512, 1024, 2048, 4096, 8192, 16384, 30720)
LENGTH_REPEATS = 5
COMPILE_RACE = "TYPE_UNSTABLE_JOIN"
NO_TRIM = "experiments.serving_concurrency.workers.QuantNoTrimWorker"
PORT = 8010


def levels(step: str) -> tuple[int, ...]:
    """Concurrency levels timed for a step: the full sweep at both ends."""
    return SWEEP if step in ("bf16", "fast") else ENDS


def swap_worker(command: list[str], worker: str) -> list[str]:
    command = command.copy()
    command[command.index("--worker-cls") + 1] = worker
    return command


def prepare(model: str, step: str, out: Path) -> tuple[list[str], dict]:
    if step not in STEPS[model]:
        raise ValueError(f"unknown step for {model}: {step}")
    if model == "4b":
        from experiments.gleipnir4b_serving_ladder.run import prepare as ladder

        return ladder(step, out)
    from experiments.release_serving.serve import prepare as release_prepare
    from experiments.release_serving.targets import RECORDED_CONTEXT, TARGETS
    from gleipnir.data.monitoring import file_hash
    from gleipnir.serving import bf16_launch

    target = TARGETS["27b"]
    release, env = release_prepare(target, "127.0.0.1", PORT, limit=RECORDED_CONTEXT)
    env["GLEIPNIR_BF16_AUDIT"] = str(out / "bf16_audit.json")
    env["GLEIPNIR_GIGATOKEN_RECEIPT"] = str(out / "frontend.json")
    if step == "fast":
        return release, env
    if step == "quant":
        return swap_worker(release, NO_TRIM), env
    paths = sorted((ROOT / "src/gleipnir/serving").glob("*.py"))
    paths += sorted(HERE.glob("*.py"))
    sources = {str(p.relative_to(ROOT)): file_hash(p) for p in paths}
    selection, command = bf16_launch.bf16_command(ROOT, target.merged_path, sources)
    command[command.index("--port") + 1] = str(PORT)
    command[command.index("--host") + 1] = "127.0.0.1"
    return command, bf16_launch.environment(ROOT, selection, out / "bf16_audit.json")


def launch(command: list[str], env: dict, name: str, out: Path, attempts: int = 3):
    """Start the server, retrying the known MXFP8 compile race."""
    from gleipnir.serving import bf16_launch

    for attempt in range(attempts):
        log = LOGS / f"{name}_attempt{attempt}.log"
        process = bf16_launch.start(
            ROOT, command, env, log, out / f"server_attempt{attempt}.json"
        )
        try:
            return process, bf16_launch.wait_ready(process, PORT), attempt
        except RuntimeError:
            bf16_launch.stop(process)
            if COMPILE_RACE not in log.read_text(errors="replace"):
                raise
            print(f"compile race on attempt {attempt}; retrying", flush=True)
    raise RuntimeError("server failed to start after retries")


async def measure(out: Path, step: str, repeats: int) -> None:
    from experiments.b200_monitor_score.run import trial
    from experiments.qwen38_27b_serving.bench import SETTINGS, workloads
    from gleipnir.data.monitoring import write_json
    from gleipnir.serving.benchmark import measurement_summary

    if SETTINGS["port"] != PORT:
        raise ValueError("benchmark port changed")
    loads = workloads()
    await trial(loads["quick"], 128, SETTINGS)
    await trial(loads["full"], 128, SETTINGS)
    for concurrency in levels(step):
        for repeat in range(repeats):
            rows, seconds = await trial(loads["full"], concurrency, SETTINGS)
            write_json(
                out / f"full_c{concurrency}_repeat{repeat}.json",
                {"rows": rows, "seconds": seconds},
            )
            summary = measurement_summary(rows, seconds)
            print(
                f"c{concurrency} repeat {repeat}: "
                f"{summary['prompt_tokens_per_second']:.0f} prompt tok/s, "
                f"p50 {1000 * summary['latency']['p50_seconds']:.1f} ms",
                flush=True,
            )
    for repeat in range(repeats):
        rows, seconds = await trial(loads["quick"], 1, SETTINGS)
        write_json(
            out / f"quick_c1_repeat{repeat}.json", {"rows": rows, "seconds": seconds}
        )


async def measure_lengths(out: Path, model_path: str) -> None:
    """Single-request latency on synthetic prompts of exact token length."""
    from transformers import AutoTokenizer

    from experiments.b200_long_context.run import synthetic
    from experiments.b200_monitor_score.run import trial
    from experiments.qwen38_27b_serving.bench import SETTINGS
    from gleipnir.data.monitoring import write_json

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    records = []
    for length in LENGTHS:
        row = synthetic(tokenizer, length)
        await trial([row], 1, SETTINGS)
        latencies = []
        for _ in range(LENGTH_REPEATS):
            values, _ = await trial([row], 1, SETTINGS)
            latencies.append(values[0]["latency_seconds"])
        records.append({"prompt_tokens": length, "latency_seconds": latencies})
        write_json(out / "lengths_c1.json", {"measurements": records})
        print(f"length {length}: {1000 * sorted(latencies)[2]:.1f} ms", flush=True)


def run(model: str, step: str, repeats: int) -> None:
    from gleipnir.data.monitoring import write_json
    from gleipnir.serving import bf16_launch

    name = f"{model}_{step}"
    out = OUT / name
    out.mkdir(parents=True, exist_ok=False)
    command, env = prepare(model, step, out)
    process, startup, attempt = launch(command, env, name, out)
    try:
        started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        asyncio.run(measure(out, step, repeats))
        if step in ("bf16", "fast"):
            asyncio.run(measure_lengths(out, command[command.index("--model") + 1]))
        audit = out / "bf16_audit.json"
        if audit.exists() and not json.loads(audit.read_text())["passed"]:
            raise ValueError("serving audit failed")
        write_json(
            out / "condition.json",
            {
                "model": model,
                "step": step,
                "worker": command[command.index("--worker-cls") + 1],
                "command": command,
                "startup_seconds": startup,
                "startup_attempt": attempt,
                "measurement_started": started,
                "levels": list(levels(step)),
                "repeats": repeats,
            },
        )
    finally:
        bf16_launch.stop(process)
    print(f"condition_complete {name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(STEPS))
    parser.add_argument("--step", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        command, _ = prepare(args.model, args.step, OUT / "_prepare")
        print(json.dumps(command, indent=1))
        return
    run(args.model, args.step, args.repeats)


if __name__ == "__main__":
    main()
