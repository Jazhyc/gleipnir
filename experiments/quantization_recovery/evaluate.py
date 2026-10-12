"""Two descriptive optimized ID passes; no held-out selection or promotion."""

import asyncio
import json
import subprocess
from pathlib import Path

import yaml

from experiments.quantization_recovery.run import (
    check,
    retire,
    server_command,
    wait_server,
)
from gleipnir.campaigns.monitoring.evaluation import (
    agreement,
    attach_inputs,
    id_metrics,
)
from gleipnir.data.monitoring import file_hash, read_rows, write_json, write_rows
from gleipnir.evaluation.http_score import score_batch


def main():
    root = Path.cwd()
    config = yaml.safe_load(
        (root / "experiments/quantization_recovery/config.yaml").read_text()
    )
    check(config)
    output = root / "results" / config["experiment"]
    reference = json.loads((output / "parity_reference.json").read_text())
    merge = json.loads((output / "merged_artifact.json").read_text())
    for path, digest in merge["files_sha256"].items():
        if file_hash(Path(merge["destination"]) / path) != digest:
            raise ValueError("candidate merged file drift")
    command, env = server_command(
        config, output, bf16=False, model=config["teacher_merge"]
    )
    write_json(output / "evaluation_command.json", {"command": command})
    with (output / "evaluation_server.log").open("x") as log:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    async def evaluate():
        await wait_server(process, config)
        lookup = {
            r["id"]: r
            for r in read_rows(root / "data" / config["experiment"] / "workload.jsonl")
        }
        canary = [lookup[i] for i in reference["ids"]]
        options = {**config["evaluation"], "concurrency": 4}
        for repeat in range(3):
            predictions, seconds = await score_batch(canary, options)
            gate = agreement(
                [p["score"] for p in predictions],
                reference["adapter"],
                reference["base"],
                config["alignment"],
            )
            write_json(
                output / f"serving_canary{repeat + 1}.json",
                {"predictions": predictions, "gate": gate, "seconds": seconds},
            )
            if not gate["passed"]:
                raise ValueError("new QAD master/serving parity failed")
        workload = json.loads(
            (root / config["inputs"]["id_workload"]["path"]).read_text()
        )
        canonical = read_rows(root / config["inputs"]["id"]["path"])
        if len(workload) != 3012 or len(canonical) != 3012:
            raise ValueError("frozen ID population drift")
        results = []
        repeated_predictions = []
        for repeat in range(config["evaluation"]["repeats"]):
            scored, durations = [], []
            for start in range(0, len(workload), 128):
                values, seconds = await score_batch(
                    workload[start : start + 128], config["evaluation"]
                )
                write_json(
                    output / f"evaluation/repeat{repeat + 1}/batches/{start:05d}.json",
                    values,
                )
                scored.extend(values)
                durations.append(seconds)
                print("QAD_ID", repeat + 1, len(scored), len(workload), flush=True)
            predictions = attach_inputs(scored, canonical)
            write_rows(output / f"evaluation/repeat{repeat + 1}/id.jsonl", predictions)
            repeated_predictions.append(predictions)
            seconds = sum(durations)
            tokens = sum(r["prompt_tokens"] for r in workload)
            result = {
                "metrics": id_metrics(predictions),
                "seconds": seconds,
                "prompt_tokens": tokens,
                "prompt_tokens_per_second": tokens / seconds,
                "batch_seconds": durations,
            }
            write_json(output / f"evaluation/repeat{repeat + 1}/summary.json", result)
            results.append(result)
        import numpy as np

        first, second = [
            np.asarray([row["score"] for row in rows]) for rows in repeated_predictions
        ]
        repeat_variation = {
            "mean_absolute_difference": float(np.abs(first - second).mean()),
            "maximum_absolute_difference": float(np.abs(first - second).max()),
            "correlation": float(np.corrcoef(first, second)[0, 1]),
            "threshold_flips_0_5": int(((first >= 0.5) != (second >= 0.5)).sum()),
        }
        write_json(
            output / "summary.json",
            {
                "status": "complete",
                "results": results,
                "repeat_variation": repeat_variation,
                "bf16_baseline": json.loads(
                    (
                        root / "results/filtered-fp4-fa4-bf16-serving01/summary.json"
                    ).read_text()
                ),
                "optimized_baseline": json.loads(
                    (
                        root / "results/filtered-fp4-fa4-optimized-id01/summary.json"
                    ).read_text()
                ),
                "promotion": False,
                "selection": "fixed_final_epoch",
                "ood_evaluated": False,
            },
        )

    try:
        asyncio.run(evaluate())
    finally:
        retire(process)


if __name__ == "__main__":
    main()
