"""Single-request latency by prompt length, from 256 to 1,048,576 tokens."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from experiments.serving.serving_concurrency.run import (
    OUT,
    PORT,
    launch,
    prepare,
    with_gigatoken,
)

NATIVE = 262144
PHASES = {
    "native": {"limit": NATIVE, "lengths": [2**i for i in range(8, 19)]},
    "yarn": {"limit": 4 * NATIVE, "lengths": [NATIVE, 2 * NATIVE, 4 * NATIVE]},
}
YARN = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": NATIVE}


def repeats(length: int) -> int:
    return 5 if length < 131072 else 3


def phase_command(command: list[str], phase: str) -> list[str]:
    """Change only the context limit and, for YaRN, the RoPE scaling."""
    limit = PHASES[phase]["limit"]
    command = command.copy()
    if command[command.index("--max-num-batched-tokens") + 1] != "32768":
        raise ValueError("chunk budget changed")
    if command[command.index("--max-model-len") + 1] not in ("32768", str(NATIVE)):
        raise ValueError("unexpected context limit")
    command[command.index("--max-model-len") + 1] = str(limit)
    index = command.index("--hf-overrides") + 1
    overrides = json.loads(command[index])
    text = overrides["text_config"]
    if text["rope_parameters"]["rope_type"] != "default" or (
        text["max_position_embeddings"] != NATIVE
    ):
        raise ValueError("served native RoPE contract changed")
    if phase == "yarn":
        text["rope_parameters"].update(YARN)
        # vLLM 0.31 follows Transformers: YaRN configs carry the scaled limit.
        text["max_position_embeddings"] = limit
        command[index] = json.dumps(overrides)
    return command


async def measure(out: Path, model_path: str, lengths: list[int]) -> None:
    from transformers import AutoTokenizer

    from experiments.b200_monitor_score.run import trial
    from experiments.serving.b200_long_context.run import synthetic
    from experiments.serving.qwen38_27b_serving.bench import SETTINGS
    from gleipnir.data.monitoring import write_json

    settings = {**SETTINGS, "timeout_seconds": 3600}
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    records = []
    for length in lengths:
        row = synthetic(tokenizer, length)
        record = {"prompt_tokens": length}
        try:
            await trial([row], 1, settings)
            latencies = []
            for _ in range(repeats(length)):
                values, _ = await trial([row], 1, settings)
                if values[0]["prompt_tokens"] != length:
                    raise ValueError("endpoint token count differs")
                latencies.append(values[0]["latency_seconds"])
            record["latency_seconds"] = latencies
            print(
                f"length {length}: {1000 * statistics.median(latencies):.1f} ms",
                flush=True,
            )
        except Exception as error:  # record the failed length and stop the phase
            record["error"] = repr(error)[:500]
            records.append(record)
            write_json(out / "lengths.json", {"measurements": records})
            print(f"length {length} failed: {record['error']}", flush=True)
            break
        records.append(record)
        write_json(out / "lengths.json", {"measurements": records})


def run(
    model: str, step: str, phase: str, session: str = "", gigatoken: bool = False
) -> None:
    from gleipnir.data.monitoring import write_json
    from gleipnir.serving import bf16_launch

    if step.split("_g")[0] not in ("bf16", "fast"):
        raise ValueError("length sweeps cover bf16 and fast only")
    name = f"{model}_{step}" + ("_giga" if gigatoken else "") + f"_{phase}"
    out = OUT / session / name
    out.mkdir(parents=True, exist_ok=False)
    command, env = prepare(model, step, out, limit=NATIVE)
    command = phase_command(command, phase)
    if gigatoken:
        env = with_gigatoken(env, out)
    log = f"{session}_{name}".lstrip("_").replace("/", "_")
    process, startup, attempt = launch(command, env, log, out)
    try:
        started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        model_path = command[command.index("--model") + 1]
        asyncio.run(measure(out, model_path, PHASES[phase]["lengths"]))
        write_json(
            out / "condition.json",
            {
                "model": model,
                "step": step,
                "phase": phase,
                "gigatoken": gigatoken,
                "worker": command[command.index("--worker-cls") + 1],
                "command": command,
                "startup_seconds": startup,
                "startup_attempt": attempt,
                "measurement_started": started,
            },
        )
    finally:
        bf16_launch.stop(process)
    print(f"condition_complete {name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("4b", "27b"))
    parser.add_argument("--step", required=True)
    parser.add_argument("--phase", required=True, choices=sorted(PHASES))
    parser.add_argument("--session", default="", help="output subdirectory")
    parser.add_argument("--gigatoken", action="store_true")
    args = parser.parse_args()
    if PORT != 8010:
        raise ValueError("benchmark port changed")
    run(args.model, args.step, args.phase, args.session, args.gigatoken)


if __name__ == "__main__":
    main()
