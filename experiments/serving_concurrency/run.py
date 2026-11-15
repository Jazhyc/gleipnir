"""Time one serving condition across client concurrency, then retire it."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "results/serving_concurrency"
LOGS = ROOT / "logs/runpod/serving_concurrency"
STEPS = {
    "4b": ("bf16", "quant", "lastquery", "route", "fast"),
    "27b": ("bf16", "quant_nograph", "quant", "lastquery", "fast"),
}
SWEEP = (128, 64, 32, 16, 8, 4, 2, 1)
ENDS = (128, 1)
COMPILE_RACE = "TYPE_UNSTABLE_JOIN"
NO_TRIM = "experiments.serving_concurrency.workers.QuantNoTrimWorker"
LAST_QUERY = "experiments.serving_concurrency.workers.QuantLastQueryWorker"
GRAPHS = "--compilation-config"
GRAPH_CAPS = (512, 1024, 2048, 4096, 8192)
PORT = 8010


def levels(step: str, ends: bool = False) -> tuple[int, ...]:
    """Concurrency levels timed for a step: the full sweep at both ends."""
    return SWEEP if step in ("bf16", "fast") and not ends else ENDS


def swap_worker(command: list[str], worker: str) -> list[str]:
    command = command.copy()
    command[command.index("--worker-cls") + 1] = worker
    return command


def without_graphs(command: list[str]) -> list[str]:
    """Drop the larger CUDA graphs; vLLM then captures up to 256 tokens."""
    index = command.index(GRAPHS)
    return command[:index] + command[index + 2 :]


def graph_sizes(command: list[str], cap: int) -> list[int]:
    """Capture sizes for the 4B up to ``cap`` tokens, padding at most 25%.

    Up to 2,048 these are the 27B release sizes; above, each power-of-two
    interval is split in four. A graph freezes the code path of its captured
    size, so the last size below each 4B switch (FP4 producer at 1,536 and
    above 4,096 rows, GDN kernel at 8,192 tokens) is added: no request is
    padded across a switch.
    """
    sizes = json.loads(command[command.index(GRAPHS) + 1])["cudagraph_capture_sizes"]
    if sizes[-1] != 2048 or cap not in GRAPH_CAPS:
        raise ValueError("release graph sizes or cap changed")
    top = 2048
    while top < cap:
        sizes += [top + top // 4 * i for i in range(1, 5)]
        top *= 2
    return sorted(n for n in {*sizes, 1535, 8191} if n <= cap)


def graph_step(step: str) -> tuple[str, int] | None:
    """``fast_g2048`` is the 4B rung ``fast`` with graphs up to 2,048 tokens."""
    match = re.fullmatch(r"(bf16|quant|lastquery|route|fast)_g(\d+)", step)
    return (match.group(1), int(match.group(2))) if match else None


def prepare(
    model: str, step: str, out: Path, limit: int | None = None
) -> tuple[list[str], dict]:
    """Command and environment of a step; ``limit`` extends the release context."""
    graphs = graph_step(step)
    if graphs is not None and model == "27b" and graphs[0] != "bf16":
        graphs = None
    if step not in STEPS[model] and graphs is None:
        raise ValueError(f"unknown step for {model}: {step}")
    from experiments.release_serving.serve import prepare as release_prepare
    from experiments.release_serving.targets import RECORDED_CONTEXT, TARGETS

    if model == "4b":
        from experiments.gleipnir4b_serving_ladder.run import prepare as ladder

        if limit is not None and step == "fast":
            return release_prepare(TARGETS["4b"], "127.0.0.1", PORT, limit=limit)
        if graphs is not None:
            command, env = ladder(graphs[0], out)
            if GRAPHS in command:
                raise ValueError("4B release command already sets graphs")
            reference, _ = release_prepare(
                TARGETS["27b"], "127.0.0.1", PORT, limit=RECORDED_CONTEXT
            )
            sizes = {"cudagraph_capture_sizes": graph_sizes(reference, graphs[1])}
            return [*command, GRAPHS, json.dumps(sizes)], env
        return ladder(step, out)
    from gleipnir.data.monitoring import file_hash
    from gleipnir.serving import bf16_launch

    target = TARGETS["27b"]
    release, env = release_prepare(
        target, "127.0.0.1", PORT, limit=limit or RECORDED_CONTEXT
    )
    env["GLEIPNIR_BF16_AUDIT"] = str(out / "bf16_audit.json")
    env["GLEIPNIR_GIGATOKEN_RECEIPT"] = str(out / "frontend.json")
    if step == "fast":
        return release, env
    if graphs is not None:  # stock BF16 with the larger graphs
        command, env = prepare(model, "bf16", out)
        sizes = {"cudagraph_capture_sizes": graph_sizes(release, graphs[1])}
        return [*command, GRAPHS, json.dumps(sizes)], env
    if step == "quant":
        return swap_worker(release, NO_TRIM), env
    if step == "quant_nograph":
        return without_graphs(swap_worker(release, NO_TRIM)), env
    if step == "lastquery":
        return swap_worker(release, LAST_QUERY), env
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


async def measure(out: Path, timed: tuple[int, ...], repeats: int) -> None:
    from experiments.b200_monitor_score.run import trial
    from experiments.qwen38_27b_serving.bench import SETTINGS, workloads
    from gleipnir.data.monitoring import write_json
    from gleipnir.serving.benchmark import measurement_summary

    if SETTINGS["port"] != PORT:
        raise ValueError("benchmark port changed")
    loads = workloads()
    await trial(loads["quick"], 128, SETTINGS)
    await trial(loads["full"], 128, SETTINGS)
    for concurrency in timed:
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


def run(
    model: str, step: str, repeats: int, session: str = "", ends: bool = False
) -> None:
    from gleipnir.data.monitoring import write_json
    from gleipnir.serving import bf16_launch

    name = f"{model}_{step}"
    out = OUT / session / name
    out.mkdir(parents=True, exist_ok=False)
    timed = levels(step, ends)
    command, env = prepare(model, step, out)
    process, startup, attempt = launch(
        command, env, f"{session}_{name}".lstrip("_"), out
    )
    try:
        started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        asyncio.run(measure(out, timed, repeats))
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
                "levels": list(timed),
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
    parser.add_argument("--session", default="", help="output subdirectory")
    parser.add_argument(
        "--ends", action="store_true", help="time concurrency 128 and 1 only"
    )
    args = parser.parse_args()
    if args.prepare_only:
        command, _ = prepare(args.model, args.step, OUT / "_prepare")
        print(json.dumps(command, indent=1))
        return
    run(args.model, args.step, args.repeats, args.session, args.ends)


if __name__ == "__main__":
    main()
