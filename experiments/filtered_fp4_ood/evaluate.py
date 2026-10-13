"""BF16 adapter parity, native audit and two fixed frozen OOD passes."""

import asyncio
import json
import subprocess
from pathlib import Path

import numpy as np
import yaml

from experiments.quantization_recovery.run import retire, server_command, wait_server
from experiments.teacher_disagreement_ood.prepare import bind_cached
from gleipnir.campaigns.monitoring.evaluation import (
    agreement,
    attach_inputs,
    id_metrics,
)
from gleipnir.data.monitoring import (
    digest,
    file_hash,
    read_rows,
    write_json,
    write_rows,
)
from gleipnir.serving.monitor_score import validate_score_response


def main():
    root = Path.cwd()
    config = yaml.safe_load(
        (root / "experiments/filtered_fp4_ood/config.yaml").read_text()
    )
    output = root / "results" / config["experiment"]
    contract = json.loads((output / "execution_contract.json").read_text())
    for path, expected in contract["files_sha256"].items():
        if file_hash(root / path) != expected:
            raise ValueError(f"frozen OOD contract drift: {path}")
    canonical, workload = (
        read_rows(root / config["canonical_ood"]),
        read_rows(root / config["ood_workload"]),
    )
    reference = json.loads((root / config["master_reference"]).read_text())
    canary = read_rows(root / config["canary"])
    command, env = server_command(
        config, output, bf16=True, model=config["teacher_merge"]
    )
    write_json(
        output / "evaluation_command.json", {"command": command, "precision": "bf16"}
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

    async def evaluate():
        from gleipnir.evaluation.http_score import score_batch

        await wait_server(process, config)
        for repeat in range(3):
            values, seconds = await score_batch(
                canary, {**config["evaluation"], "concurrency": 4}
            )
            gate = agreement(
                [r["score"] for r in values],
                reference["adapter"],
                reference["base"],
                config["alignment"],
            )
            write_json(
                output / f"serving_canary{repeat + 1}.json",
                {"gate": gate, "predictions": values, "seconds": seconds},
            )
            if not gate["passed"]:
                raise ValueError("new BF16 serving parity failed")
        if not json.loads((output / "bf16_audit.json").read_text())["passed"]:
            raise ValueError("native BF16 audit failed")
        results, scores = [], []
        for repeat in range(1, config["repeats"] + 1):
            values, durations = [], []
            for start in range(0, len(workload), 128):
                batch, seconds = await score_batch(
                    workload[start : start + 128], config["evaluation"]
                )
                write_json(
                    output / f"evaluation/repeat{repeat}/batches/{start:05d}.json",
                    batch,
                )
                values.extend(batch)
                durations.append(seconds)
                print("FP4_OOD", repeat, len(values), len(workload), flush=True)
            predictions = attach_inputs(values, canonical)
            for value, row in zip(predictions, workload, strict=True):
                if value["prompt_sha256"] != row["prompt_sha256"] or row[
                    "prompt_sha256"
                ] != digest(row["prompt"]):
                    raise ValueError("frozen rendered prompt identity mismatch")
                validate_score_response(value, row["prompt_tokens"])
            write_rows(output / f"evaluation/repeat{repeat}/ood.jsonl", predictions)
            tokens = sum(r["prompt_tokens"] for r in workload)
            result = {
                "metrics": id_metrics(predictions),
                "seconds": sum(durations),
                "prompt_tokens": tokens,
                "prompt_tokens_per_second": tokens / sum(durations),
            }
            write_json(output / f"evaluation/repeat{repeat}/summary.json", result)
            results.append(result)
            scores.append(np.asarray([r["score"] for r in predictions]))
        comparisons = {}
        for name, path in config["cached_references"].items():
            rows = read_rows(root / path)
            if name == "filtered_sdpa":
                rows = attach_inputs(rows, canonical)
                for value, row in zip(rows, workload, strict=True):
                    if value["prompt_sha256"] != row["prompt_sha256"]:
                        raise ValueError("cached strict control rendering drift")
            else:
                rows = bind_cached(rows, canonical)
            comparisons[name] = id_metrics(rows)
        a, b = scores
        write_json(
            output / "summary.json",
            {
                "status": "complete",
                "precision": "bf16",
                "results": results,
                "cached_references": comparisons,
                "primary_reference": "filtered_sdpa",
                "repeat_variation": {
                    "mean_absolute_difference": float(np.abs(a - b).mean()),
                    "maximum_absolute_difference": float(np.abs(a - b).max()),
                    "correlation": float(np.corrcoef(a, b)[0, 1]),
                    "threshold_flips_0_5": int(((a >= 0.5) != (b >= 0.5)).sum()),
                },
                "promotion": False,
                "selection": "fixed_original_adapter_no_ood_tuning",
            },
        )

    try:
        asyncio.run(evaluate())
    finally:
        retire(process)


if __name__ == "__main__":
    main()
