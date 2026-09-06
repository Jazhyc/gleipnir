"""Resume one prefix adapter over two independent GPU evaluation workers."""

import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from experiments.adapter_capacity_scaling.run_lambda import runtime_environment
from experiments.monitoring_duration.run import summarize as summarize_campaign
from experiments.tool_trajectory_monitoring.benchmark_distilled_ood import (
    sha256_file,
    validate_inputs,
)
from experiments.tool_trajectory_monitoring.benchmark_gpt_oss_ood import summarize
from gleipnir.evaluation_shards import merge_predictions, partition_pending
from gleipnir.monitoring_systems_screen import atomic_write_json, atomic_write_jsonl


def main() -> None:
    root = Path("results/monitoring_prefix_training")
    job = "prefix-w050-lr2em05-seed0"
    output = root / "id_evaluation/4b/adapters" / job
    work = root / "evaluation_restart_sharded"
    memory = [
        int(v)
        for v in subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        ).splitlines()
    ]
    if len(memory) != 2 or max(memory) > 1024:
        raise RuntimeError("two idle GPUs required; do not interrupt other workloads")
    config_path = root / "id_benchmark.json"
    config = json.loads(config_path.read_text())
    rows = validate_inputs(config)
    digest = sha256_file(config_path)
    work.mkdir(exist_ok=False)
    snapshot = work / "resume_snapshot.jsonl"
    shutil.copy2(output / "predictions.jsonl", snapshot)
    saved = [json.loads(s) for s in snapshot.read_text().splitlines()]
    assignments = [
        partition_pending(rows, saved, digest, 2, i, config["engine"]["batch_rows"])
        for i in range(2)
    ]
    if any(not a for a in assignments):
        raise ValueError("not enough pending batches for two workers")
    status = json.loads((root / "status.json").read_text())
    status.update(
        state="running",
        phase="id_evaluation_sharded",
        evaluation_restart={
            "saved_rows": len(saved),
            "shard_rows": [len(a) for a in assignments],
            "started_at_unix": time.time(),
        },
    )
    atomic_write_json(root / "status.json", status)

    def worker(index: int) -> dict:
        directory = work / f"gpu{index}"
        with (work / f"gpu{index}.log").open("x") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "experiments.tool_trajectory_monitoring.benchmark_distilled_ood",
                    "--config",
                    str(config_path),
                    "--model-size",
                    "4b",
                    "--output-root",
                    str(directory),
                    "--only-job",
                    job,
                    "--shard-count",
                    "2",
                    "--shard-index",
                    str(index),
                    "--resume-snapshot",
                    str(snapshot),
                ],
                env=runtime_environment(str(index)),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        return json.loads((directory / "4b/adapters" / job / "result.json").read_text())

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(worker, range(2)))
        parts = [saved]
        for i in range(2):
            p = work / f"gpu{i}/4b/adapters" / job / "predictions.jsonl"
            predictions = [json.loads(s) for s in p.read_text().splitlines()]
            if {str(r["id"]) for r in predictions} != {
                str(r["id"]) for r in assignments[i]
            }:
                raise ValueError("shard assignment drift")
            parts.append(predictions)
        merged = merge_predictions(rows, parts, digest)
        atomic_write_jsonl(output / "predictions.jsonl", merged)
        result = {
            **results[0],
            **summarize(merged),
            "evaluation_shard": None,
            "runtime": {
                "execution": "two_gpu_sharded_resume",
                "saved_rows": len(saved),
                "shards": [r["runtime"] for r in results],
            },
            "merge_provenance": {
                "snapshot_sha256": sha256_file(snapshot),
                "shard_rows": [len(a) for a in assignments],
            },
        }
        atomic_write_json(output / "result.json", result)
        jobs = [json.loads(s) for s in (root / "jobs.jsonl").read_text().splitlines()]
        summarize_campaign(
            root, jobs, json.loads((root / "resolved_config.json").read_text())
        )
        status.update(state="complete", phase="complete", completed_at_unix=time.time())
    except BaseException as error:
        status.update(state="failed", error=repr(error))
        raise
    finally:
        status["updated_at_unix"] = time.time()
        atomic_write_json(root / "status.json", status)


if __name__ == "__main__":
    main()
