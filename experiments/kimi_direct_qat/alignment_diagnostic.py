"""Repeat the fresh-base serving canary without updates or changing failed gates."""

import asyncio
import json
import subprocess
from pathlib import Path

import yaml

from experiments.quantization_recovery.run import retire, server_command, wait_server
from gleipnir.campaigns.monitoring.evaluation import agreement
from gleipnir.data.monitoring import read_rows, write_json
from gleipnir.evaluation.http_score import score_batch


def main():
    root = Path.cwd()
    config = yaml.safe_load(
        (root / "experiments/kimi_direct_qat/config.yaml").read_text()
    )
    campaign = root / "results" / config["experiment"]
    output = campaign / "alignment_diagnostic"
    output.mkdir(exist_ok=True)
    failed = json.loads((campaign / "alignment.json").read_text())
    if failed["passed"] or (output / "result.json").exists():
        raise ValueError("diagnostic requires an unmodified failed alignment")
    workload = read_rows(
        root / "data" / config["teacher_experiment"] / "workload.jsonl"
    )
    indexed = {row["id"]: row for row in workload}
    canary = [indexed[identity] for identity in failed["ids"]]
    original = json.loads(
        (
            root
            / "results/filtered-fp4-fa4-baseline01/4b/monitor/parity_reference.json"
        ).read_text()
    )
    command, env = server_command(
        config, output, bf16=False, model=root / config["base_model"]
    )
    with (output / "server.log").open("x") as stream:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    async def score():
        await wait_server(process, config)
        results = []
        for repeat in range(3):
            predictions, seconds = await score_batch(
                canary, {**config["evaluation"], "concurrency": 4}
            )
            gate = agreement(
                failed["scores"],
                [p["score"] for p in predictions],
                original["base"],
                config["initial_alignment"],
            )
            row = {"repeat": repeat + 1, "gate": gate, "seconds": seconds}
            write_json(
                output / f"repeat{repeat + 1}.json", {**row, "predictions": predictions}
            )
            print("fresh_base_alignment_repeat", json.dumps(row), flush=True)
            results.append(row)
        write_json(output / "result.json", {"results": results, "optimizer_updates": 0})

    try:
        asyncio.run(score())
    finally:
        retire(process)


if __name__ == "__main__":
    main()
