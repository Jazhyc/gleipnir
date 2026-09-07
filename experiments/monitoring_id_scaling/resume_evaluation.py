"""Resume completed scaling adapters using two persistent inference engines."""

import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from experiments.adapter_capacity_scaling.run_lambda import runtime_environment
from experiments.monitoring_duration.run import summarize, validate_completed
from experiments.monitoring_id_scaling.run import ROOT, make_jobs, write_curve
from experiments.monitoring_lr_sweep.run_lambda import gpu_training_environment
from experiments.monitoring_objective_ablation.run_lambda import run_serving_parity
from experiments.tool_trajectory_monitoring.benchmark_distilled_ood import (
    validate_config,
    validate_inputs,
    validate_jobs,
)
from gleipnir.evaluation_lanes import adapter_lanes
from gleipnir.monitoring_systems_screen import (
    atomic_write_json,
    read_jsonl,
    sha256_file,
)
from gleipnir.qwen35_fast_training import ensure_qwen35_long_trajectory_kernels


def main() -> None:
    root = ROOT
    config = json.loads((root / "resolved_config.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    # User-authorized evaluation scheduling refactor; original manifest is retained.
    code_transition = {
        "path": "/home/ubuntu/gleipnir/experiments/monitoring_id_scaling/run.py",
        "before": "40c98925277a20a33778c1eb333cf9b31a1f810976efabd1be68488806d02385",
        "after": "a45965ec838506a56638b25bc5d0f2d7a7f932c6601f8a8596f2dc2a0fc0361d",
    }
    for path, digest in manifest["files"].items():
        if path == code_transition["path"] and digest == code_transition["before"]:
            digest = code_transition["after"]
        if sha256_file(Path(path)) != digest:
            raise ValueError(f"frozen contract drift: {path}")
    jobs = read_jsonl(root / "jobs.jsonl")
    if jobs != make_jobs(config):
        raise ValueError("job reconstruction drift")
    for job in jobs:
        validate_completed(job)
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    if processes:
        raise RuntimeError(f"GPUs must be idle before takeover: {processes}")
    evaluation_path = root / "id_benchmark.json"
    evaluation = json.loads(evaluation_path.read_text())
    validate_config(evaluation)
    validate_inputs(evaluation)
    validate_jobs(evaluation, "4b")
    lanes = adapter_lanes([j["job_name"] for j in jobs])
    status = json.loads((root / "status.json").read_text())
    status.update(
        state="running",
        phase="serving_parity",
        active_jobs=[],
        completed_jobs=[j["job_name"] for j in jobs],
        evaluation_lanes=lanes,
    )
    for name in status["completed_jobs"]:
        status["job_status"][name]["state"] = "complete"
    atomic_write_json(
        root / "evaluation_split.json",
        {
            "lanes": lanes,
            "config_sha256": sha256_file(evaluation_path),
            "started_at_unix": time.time(),
            "preserved_existing_predictions": True,
            "execution": "one persistent TP1 engine per GPU; disjoint adapters",
            "authorized_code_transition": code_transition,
        },
    )

    def worker(gpu: int) -> None:
        command = [
            sys.executable,
            "-u",
            "-m",
            "experiments.tool_trajectory_monitoring.benchmark_distilled_ood",
            "--config",
            str(evaluation_path),
            "--model-size",
            "4b",
            "--output-root",
            str(root / "id_evaluation"),
        ]
        for name in lanes[gpu]:
            command.extend(["--only-job", name])
        with (root / f"evaluation_gpu{gpu}.log").open("a") as log:
            subprocess.run(
                command,
                env=runtime_environment(str(gpu)),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

    try:
        atomic_write_json(root / "status.json", status)
        fast = ensure_qwen35_long_trajectory_kernels()
        run_serving_parity(
            evaluation_path,
            root,
            gpu_training_environment(fast, 0),
            runtime_environment("0"),
        )
        status.update(phase="id_evaluation_parallel")
        atomic_write_json(root / "status.json", status)
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(worker, range(len(lanes))))
        summarize(root, jobs, config)
        write_curve(root, config)
        status.update(state="complete", phase="complete", completed_at_unix=time.time())
    except BaseException as error:
        status.update(state="failed", error=repr(error))
        raise
    finally:
        status["updated_at_unix"] = time.time()
        atomic_write_json(root / "status.json", status)
        queue = json.loads((root / "queue_status.json").read_text())
        queue.update(state=status["state"], execution="two_gpu_adapter_split")
        atomic_write_json(root / "queue_status.json", queue)


if __name__ == "__main__":
    main()
