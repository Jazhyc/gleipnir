"""Prepare and execute a bounded duration screen using the existing training stack."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from experiments.adapter_capacity_scaling.run_lambda import runtime_environment
from experiments.monitoring_lr_sweep.core import validate_training_metadata
from experiments.monitoring_lr_sweep.prepare import (
    DEFAULT_ID_INPUT,
    inspect_training_inputs,
    validate_id_separation,
)
from experiments.monitoring_lr_sweep.prepare import (
    make_jobs as make_lr_jobs,
)
from experiments.monitoring_lr_sweep.run_lambda import gpu_training_environment
from experiments.monitoring_objective_ablation.run_lambda import run_serving_parity
from experiments.tool_trajectory_monitoring.benchmark_distilled_ood import (
    validate_config,
    validate_inputs,
)
from experiments.tool_trajectory_monitoring.benchmark_distilled_ood import (
    validate_jobs as validate_evaluation_jobs,
)
from experiments.tool_trajectory_monitoring.run_distillation_lambda import (
    run_training_job,
    verify_job_inputs,
)
from gleipnir.campaign_status import CampaignStatus
from gleipnir.monitoring_systems_screen import (
    atomic_write_json,
    atomic_write_jsonl,
    read_jsonl,
    sha256_file,
)
from gleipnir.qwen35_fast_training import ensure_qwen35_long_trajectory_kernels

DEFAULT_ROOT = Path("results/monitoring_duration")


def make_jobs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Change only duration/LR and artifact identity from the selected recipe."""
    if (
        config["learning_rates"] != [1e-5, 2e-5]
        or config["epochs"] != 2
        or config["seed"] != 0
        or config["strict_ood_consulted"] is not False
    ):
        raise ValueError("duration design differs from the two authorized cells")
    jobs = []
    for source in make_lr_jobs(Path(config["data_dir"]), Path(config["result_dir"])):
        if source["learning_rate"] not in config["learning_rates"]:
            continue
        name = source["job_name"] + "-epochs2"
        output = Path(config["result_dir"]) / "runs" / name
        jobs.append(
            {
                **source,
                "job_name": name,
                "design_role": "two_epoch_duration_candidate",
                "num_train_epochs": 2.0,
                "save_steps": 544,
                "completion_loss_weight": 0.0,
                "mil_loss_weight": 0.0,
                "output_dir": str(output),
                "causal_adapter_dir": str(output / "causal_adapter"),
                "model_dir": str(output / "model"),
            }
        )
    return jobs


def prepare() -> None:
    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).parent.resolve())
    ):
        config = OmegaConf.to_container(compose(config_name="config"), resolve=True)
    root = Path(config["result_dir"])
    if (root / "manifest.json").exists():
        raise ValueError("campaign already prepared; reuse its frozen contract")
    data = Path(config["data_dir"])
    audit, selection = inspect_training_inputs(
        data / "student_rows.jsonl", data / "soft_targets.jsonl"
    )
    heldout = validate_id_separation(DEFAULT_ID_INPUT, audit.pop("trajectory_hashes"))
    jobs = make_jobs(config)
    atomic_write_jsonl(root / "jobs.jsonl", jobs)
    atomic_write_jsonl(root / "selections/preflight-longest-32.jsonl", selection)
    evaluation = json.loads(Path(config["evaluation_template"]).read_text())
    evaluation.update(
        campaign_id=config["campaign_id"],
        hypothesis="Two-epoch soft-only duration screen",
    )
    evaluation["scope"]["selection_rule"] = (
        "Evaluate the two frozen final checkpoints only; "
        "no intermediate selection or strict OOD."
    )
    evaluation["model_groups"]["4b"].update(
        jobs=str(root / "jobs.jsonl"),
        jobs_sha256=sha256_file(root / "jobs.jsonl"),
        expected_jobs=[job["job_name"] for job in jobs],
        parity_job=jobs[1]["job_name"],
    )
    atomic_write_json(root / "id_benchmark.json", evaluation)
    atomic_write_json(root / "resolved_config.json", config)
    validate_config(evaluation)
    validate_inputs(evaluation)
    validate_evaluation_jobs(evaluation, "4b")
    paths = [
        root / name
        for name in (
            "jobs.jsonl",
            "id_benchmark.json",
            "resolved_config.json",
            "selections/preflight-longest-32.jsonl",
        )
    ]
    paths.append(Path(config["baseline_result"]))
    atomic_write_json(
        root / "manifest.json",
        {
            "training": audit,
            "held_out_id": heldout,
            "authoring_sha256": sha256_file(Path(__file__).with_name("config.yaml")),
            "files": {str(path): sha256_file(path) for path in paths},
        },
    )


def validate_completed(job: dict[str, Any], *, preflight: bool = False) -> None:
    metadata = validate_training_metadata(
        Path(job["causal_adapter_dir"]) / "training_metadata.json",
        job["learning_rate"],
        expected_steps=1 if preflight else 544,
        require_canary=preflight,
    )
    losses = metadata["losses"]
    if any(
        float(losses.get(key, 0)) != 0
        for key in (
            "completion_weight",
            "direct_weight",
            "pairwise_weight",
            "mil_weight",
        )
    ):
        raise ValueError("unexpected auxiliary objective")
    if float(losses.get("soft_weight", -1)) != 1 or losses.get("soft_type") != "bce":
        raise ValueError("soft BCE identity drift")
    if not math.isfinite(float(metadata["train_metrics"]["train_loss"])):
        raise ValueError("nonfinite training loss")


def summarize(root: Path, jobs: list[dict[str, Any]], config: dict[str, Any]) -> None:
    baseline = json.loads(Path(config["baseline_result"]).read_text())["metrics"][
        "macro"
    ]
    control = baseline["macro"]
    sources = {row["group"]: row["pauroc_at_20"] for row in baseline["groups"]}
    rows = []
    for job in jobs:
        result = json.loads(
            (
                root / "id_evaluation/4b/adapters" / job["job_name"] / "result.json"
            ).read_text()
        )
        if result["rows"] != 3012 or result["job_name"] != job["job_name"]:
            raise ValueError("incomplete final evaluation")
        metrics = result["metrics"]["macro"]
        macro = metrics["macro"]
        gain = macro["pauroc_at_20"] - control["pauroc_at_20"]
        source_deltas = {
            row["group"]: row["pauroc_at_20"] - sources[row["group"]]
            for row in metrics["groups"]
        }
        rule = config["selection"]
        rows.append(
            {
                "job_name": job["job_name"],
                "metrics": metrics,
                "control_pauroc_gain": gain,
                "source_pauroc_deltas": source_deltas,
                "passes_exploratory_gate": gain >= rule["min_pauroc_gain"]
                and min(source_deltas.values()) >= -rule["max_source_regression"]
                and macro["brier"] - control["brier"] <= rule["max_brier_regression"],
            }
        )
    atomic_write_json(
        root / "summary.json",
        {
            "rows": rows,
            "baseline": baseline,
            "strict_ood_consulted": False,
            "interpretation": (
                "Fixed-endpoint single-seed ID development; no automatic promotion."
            ),
        },
    )


def execute(root: Path, revision: str | None) -> None:
    manifest = json.loads((root / "manifest.json").read_text())
    for name, checksum in manifest["files"].items():
        if sha256_file(Path(name)) != checksum:
            raise ValueError(f"frozen contract drift: {name}")
    config = json.loads((root / "resolved_config.json").read_text())
    jobs_path = root / "jobs.jsonl"
    jobs = read_jsonl(jobs_path)
    if jobs != make_jobs(config):
        raise ValueError("job reconstruction drift")
    verify_job_inputs(jobs)
    evaluation_path = root / "id_benchmark.json"
    evaluation = json.loads(evaluation_path.read_text())
    validate_config(evaluation)
    validate_inputs(evaluation)
    validate_evaluation_jobs(evaluation, "4b")
    status = CampaignStatus(root / "status.json", jobs, revision)
    logs = Path("logs/lambda/monitoring_duration")
    logs.mkdir(parents=True, exist_ok=True)
    serving = runtime_environment("0")
    os.environ.update(serving)
    try:
        status.update(phase="kernel_preflight")
        fast = ensure_qwen35_long_trajectory_kernels()
        if revision:
            fast["GLEIPNIR_COMMIT"] = serving["GLEIPNIR_COMMIT"] = revision
        output = root / "preflight"
        selection = root / "selections/preflight-longest-32.jsonl"
        preflight = {
            **jobs[1],
            "job_name": "preflight-longest32",
            "train_rows": 32,
            "max_steps": 1,
            "num_train_epochs": -1,
            "save_steps": 1,
            "selection_manifest": str(selection),
            "selection_sha256": sha256_file(selection),
            "selective_torch_compile_canary_tokens": 2048,
            "output_dir": str(output),
            "causal_adapter_dir": str(output / "causal_adapter"),
            "model_dir": str(output / "model"),
        }
        atomic_write_jsonl(root / "preflight_jobs.jsonl", [preflight])
        status.update(phase="longest_sequence_preflight", preflight="running")
        run_training_job(
            root / "preflight_jobs.jsonl",
            preflight["job_name"],
            gpu_training_environment(fast, 0),
            preflight=True,
        )
        validate_completed(preflight, preflight=True)
        status.update(phase="training", preflight="passed")

        def lane(job: dict[str, Any], gpu: int) -> None:
            with status.job(job["job_name"], gpu=gpu):
                command = [
                    sys.executable,
                    "-u",
                    "-m",
                    "experiments.tool_trajectory_monitoring.run_distillation_train",
                    "--jobs",
                    str(jobs_path),
                    "--job-name",
                    job["job_name"],
                    "--allow-non-scaling-job",
                ]
                with (logs / f"gpu{gpu}.log").open("a") as log:
                    subprocess.run(
                        command,
                        env=gpu_training_environment(fast, gpu),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                validate_completed(job)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(lane, job, gpu) for gpu, job in enumerate(jobs)]
            for future in futures:
                future.result()
        status.update(phase="serving_parity")
        run_serving_parity(
            evaluation_path, root, gpu_training_environment(fast, 0), serving
        )
        status.update(phase="id_evaluation")
        subprocess.run(
            [
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
            ],
            env=serving,
            check=True,
        )
        summarize(root, jobs, config)
        status.update(state="complete", phase="complete", completed_at_unix=time.time())
    except BaseException as error:
        status.update(state="failed", error=repr(error), failed_at_unix=time.time())
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--revision", default=os.environ.get("GLEIPNIR_COMMIT"))
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare()
    else:
        execute(args.result_dir, args.revision)


if __name__ == "__main__":
    main()
