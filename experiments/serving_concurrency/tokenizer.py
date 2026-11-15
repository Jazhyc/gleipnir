"""Time one serving step with the native Gigatoken and the HF tokenizer.

One server; ``GLEIPNIR_GIGATOKEN_AB`` lets SIGUSR1 switch the encoder between
drained passes. Each round times the benchmark at concurrency 128 and 1
with the native encoder, then with the tokenizer of Transformers.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import time
from pathlib import Path

from experiments.serving_concurrency.run import ENDS, OUT, launch, prepare


def switch(receipt: Path, mode: str) -> dict:
    """Toggle the encoder of the API server and wait until it reports ``mode``."""
    state = json.loads(receipt.read_text())
    if state["ab_control"]["mode"] != mode:
        os.kill(state["pid"], signal.SIGUSR1)
        for _ in range(100):
            time.sleep(0.1)
            state = json.loads(receipt.read_text())
            if state["ab_control"]["mode"] == mode:
                break
        else:
            raise RuntimeError(f"encoder did not switch to {mode}")
    return state["ab_control"]


async def measure(out: Path, receipt: Path, rounds: int) -> None:
    from experiments.b200_monitor_score.run import trial
    from experiments.qwen38_27b_serving.bench import SETTINGS, workloads
    from gleipnir.data.monitoring import write_json

    loads = workloads()
    await trial(loads["quick"], 128, SETTINGS)
    await trial(loads["full"], 128, SETTINGS)
    for repeat in range(rounds):
        for mode in ("native", "hf"):
            switch(receipt, mode)
            for concurrency in ENDS:
                rows, seconds = await trial(loads["full"], concurrency, SETTINGS)
                write_json(
                    out / f"{mode}_full_c{concurrency}_repeat{repeat}.json",
                    {"rows": rows, "seconds": seconds},
                )
                tokens = sum(r["prompt_tokens"] for r in rows)
                print(
                    f"{mode} c{concurrency} repeat {repeat}: "
                    f"{tokens / seconds:.0f} prompt tok/s",
                    flush=True,
                )
    write_json(out / "encoder.json", switch(receipt, "native"))


def run(model: str, step: str, session: str, rounds: int) -> None:
    from gleipnir.data.monitoring import write_json
    from gleipnir.serving import bf16_launch

    name = f"tokenizer_{model}_{step}"
    out = OUT / session / name
    out.mkdir(parents=True, exist_ok=False)
    command, env = prepare(model, step, out)
    env["GLEIPNIR_GIGATOKEN_AB"] = "1"
    receipt = Path(env["GLEIPNIR_GIGATOKEN_RECEIPT"])
    log = f"{session}_{name}".lstrip("_").replace("/", "_")
    process, startup, attempt = launch(command, env, log, out)
    try:
        asyncio.run(measure(out, receipt, rounds))
        write_json(
            out / "condition.json",
            {
                "model": model,
                "step": step,
                "command": command,
                "rounds": rounds,
                "startup_seconds": startup,
                "startup_attempt": attempt,
            },
        )
    finally:
        bf16_launch.stop(process)
    print(f"tokenizer_complete {name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("4b", "27b"))
    parser.add_argument("--step", required=True)
    parser.add_argument("--session", default="", help="output subdirectory")
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    run(args.model, args.step, args.session, args.rounds)


if __name__ == "__main__":
    main()
