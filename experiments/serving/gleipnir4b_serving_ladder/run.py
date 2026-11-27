"""Measure one cumulative serving rung of Gleipnir 4B, then retire the server."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()
)
HERE = Path(__file__).resolve().parent
OUT = ROOT / "results/gleipnir4b_serving_ladder"
LOGS = ROOT / "logs/runpod/gleipnir4b_serving_ladder"
RUNGS = ("bf16", "quant", "lastquery", "route", "fast")
PORT = 8010  # the port of experiments.serving.qwen38_27b_serving.bench.SETTINGS


def layered_command(root: Path, rung: str) -> list[str]:
    """The selected recipe's command with worker layers up to ``rung``."""
    from gleipnir.serving.reference import (
        apply_last_query,
        apply_worker_layer,
        selected_serving_default,
    )

    if rung not in RUNGS[1:]:
        raise ValueError(f"not an optimized rung: {rung}")
    selection, command = selected_serving_default(root, last_query=False)
    bindings = selection["artifact_bindings"]
    depth = RUNGS.index(rung)
    if depth >= 2:
        apply_last_query(root, command, selection["last_query_attention"], bindings)
    if depth >= 3:
        route = selection["gdn_route"]
        gate = {"cp_min_tokens": int(route["cp_min_tokens"])}
        apply_worker_layer(root, command, route, bindings, ("gdn_route", gate))
    if depth >= 4:
        apply_worker_layer(root, command, selection["gdn_fast"], bindings)
    return command


def retarget(command: list[str], release: list[str]) -> list[str]:
    """Point a layered command at the release command's model and socket.

    The recorded ``serving_condition.merged_model`` label follows the model.

    Every other argument must already agree with the release command, apart
    from the worker class and the additional config that the layers change.
    """
    command = command.copy()
    command[0] = release[0]
    for flag in ("--model", "--host", "--port"):
        command[command.index(flag) + 1] = release[release.index(flag) + 1]
    index = command.index("--additional-config") + 1
    additional = json.loads(command[index])
    additional["serving_condition"]["merged_model"] = json.loads(
        release[release.index("--additional-config") + 1]
    )["serving_condition"]["merged_model"]
    command[index] = json.dumps(additional, sort_keys=True)
    skip = {command.index(f) + 1 for f in ("--worker-cls", "--additional-config")}
    differing = [
        (a, b)
        for i, (a, b) in enumerate(zip(command, release, strict=True))
        if i not in skip and a != b
    ]
    if differing:
        raise ValueError(f"layered and release commands differ: {differing}")
    return command


def prepare(rung: str, out: Path | None = None) -> tuple[list[str], dict]:
    from experiments.release_serving.serve import prepare as release_prepare
    from experiments.release_serving.targets import RECORDED_CONTEXT, TARGETS
    from gleipnir.data.monitoring import file_hash
    from gleipnir.serving import bf16_launch

    target = TARGETS["4b"]
    release, env = release_prepare(target, "127.0.0.1", PORT, limit=RECORDED_CONTEXT)
    if rung == "bf16":
        paths = sorted((ROOT / "src/gleipnir/serving").glob("*.py"))
        paths += sorted(HERE.glob("*.py"))
        sources = {str(p.relative_to(ROOT)): file_hash(p) for p in paths}
        selection, command = bf16_launch.bf16_command(ROOT, target.merged_path, sources)
        command[command.index("--port") + 1] = str(PORT)
        command[command.index("--host") + 1] = "127.0.0.1"
        audit = (out or OUT / rung) / "bf16_audit.json"
        return command, bf16_launch.environment(ROOT, selection, audit)
    command = retarget(layered_command(ROOT, rung), release)
    if rung == "fast" and command != release:
        raise ValueError("fully layered command differs from the release command")
    return command, env


def run(rung: str, repeats: int, name: str | None = None) -> None:
    from experiments.serving.qwen38_27b_serving import bench
    from gleipnir.data.monitoring import write_json
    from gleipnir.serving import bf16_launch

    if bench.SETTINGS["port"] != PORT:
        raise ValueError("benchmark port changed")
    name = name or rung
    out = OUT / name
    out.mkdir(parents=True, exist_ok=False)
    command, env = prepare(rung, out)
    process = bf16_launch.start(
        ROOT, command, env, LOGS / f"{name}.log", out / "server.json"
    )
    try:
        startup = bf16_launch.wait_ready(process, PORT)
        asyncio.run(bench.measure(out, repeats))
        if rung == "bf16":
            audit = json.loads((out / "bf16_audit.json").read_text())
            if not audit["passed"]:
                raise ValueError("BF16 audit failed")
        write_json(
            out / "condition.json",
            {
                "rung": rung,
                "worker": command[command.index("--worker-cls") + 1],
                "startup_seconds": startup,
                "repeats": repeats,
                "model": command[command.index("--model") + 1],
            },
        )
    finally:
        bf16_launch.stop(process)
    print(f"rung_complete {name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rung", required=True, choices=RUNGS)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--name", help="output directory name; default the rung")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        command, _ = prepare(args.rung)
        json.dump(command, sys.stdout, indent=1)
        return
    run(args.rung, args.repeats, args.name)


if __name__ == "__main__":
    main()
