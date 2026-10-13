"""Adapter parity, native audit and fixed frozen OOD serving comparisons."""

import asyncio
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from experiments.filtered_fp4_ood.run import load_config
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


def repeat_variation(scores: list[np.ndarray]) -> dict | None:
    """Leave serving variability unmeasured when only one pass is requested."""
    if len(scores) == 1:
        return None
    if len(scores) != 2:
        raise ValueError("expected one or two fixed passes")
    a, b = scores
    return {
        "mean_absolute_difference": float(np.abs(a - b).mean()),
        "maximum_absolute_difference": float(np.abs(a - b).max()),
        "correlation": float(np.corrcoef(a, b)[0, 1]),
        "threshold_flips_0_5": int(((a >= 0.5) != (b >= 0.5)).sum()),
    }


def evaluate_condition(config, condition):
    root = Path.cwd()
    parent = root / "results" / config["experiment"]
    output = parent if len(config["conditions"]) == 1 else parent / condition
    output.mkdir(parents=True, exist_ok=True)
    contract = json.loads((parent / "execution_contract.json").read_text())
    for path, expected in contract["files_sha256"].items():
        if file_hash(root / path) != expected:
            raise ValueError(f"frozen OOD contract drift: {path}")
    canonical, workload = (
        read_rows(root / config["canonical_ood"]),
        read_rows(root / config["ood_workload"]),
    )
    reference = json.loads((root / config["master_reference"]).read_text())
    canary = read_rows(root / config["canary"])
    if (
        "master_sha256" in reference
        and reference["master_sha256"] != config["initial_adapter_sha256"]
    ):
        raise ValueError("master reference identity drift")
    if (
        "prompt_sha256" in reference
        and [r["prompt_sha256"] for r in canary] != reference["prompt_sha256"]
    ):
        raise ValueError("master canary rendering/order drift")
    command, env = server_command(
        config, output, bf16=condition == "bf16", model=config["teacher_merge"]
    )
    write_json(
        output / "evaluation_command.json", {"command": command, "precision": condition}
    )
    launched_at = time.time()
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
                raise ValueError(f"new {condition} serving parity failed")
        if condition == "bf16":
            if not json.loads((output / "bf16_audit.json").read_text())["passed"]:
                raise ValueError("native BF16 audit failed")
        else:
            path = (
                root
                / "results/b200_attention_gdn_serving/native_attention_projections.json"
            )
            audit = json.loads(path.read_text())
            if (
                path.stat().st_mtime < launched_at
                or not audit["passed"]
                or audit["precision"] != "fp8"
            ):
                raise ValueError("fresh optimized attention dispatch audit failed")
            shutil.copy2(path, output / "native_attention_projections.json")
        results, scores = [], []
        for repeat in range(
            1, config.get("condition_repeats", {}).get(condition, config["repeats"]) + 1
        ):
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
                print("OOD", condition, repeat, len(values), len(workload), flush=True)
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
            if name == "filtered_sdpa" or name.startswith("fp4_bf16"):
                rows = attach_inputs(rows, canonical)
                for value, row in zip(rows, workload, strict=True):
                    if value["prompt_sha256"] != row["prompt_sha256"]:
                        raise ValueError("cached strict control rendering drift")
            else:
                rows = bind_cached(rows, canonical)
            comparisons[name] = id_metrics(rows)
        variation = repeat_variation(scores)
        write_json(
            output / "summary.json",
            {
                "status": "complete",
                "precision": condition,
                "results": results,
                "cached_references": comparisons,
                "primary_reference": "filtered_sdpa",
                "repeat_variation": variation,
                "promotion": False,
                "selection": config.get(
                    "selection", "fixed_original_adapter_no_ood_tuning"
                ),
            },
        )

    try:
        asyncio.run(evaluate())
    finally:
        retire(process)


def main():
    _, config = load_config()
    for condition in config["conditions"]:
        evaluate_condition(config, condition)
    if len(config["conditions"]) > 1:
        output = Path.cwd() / "results" / config["experiment"]
        write_json(
            output / "summary.json",
            {
                "status": "complete",
                "conditions": {
                    c: json.loads((output / c / "summary.json").read_text())
                    for c in config["conditions"]
                },
                "promotion": False,
            },
        )


if __name__ == "__main__":
    main()
