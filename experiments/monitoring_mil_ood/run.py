"""Evaluate the fixed ID-selected MIL3 checkpoint; never train or select on OOD."""

import json
import os
import subprocess
import sys
from pathlib import Path

from experiments.adapter_capacity_scaling.run_lambda import runtime_environment
from experiments.tool_trajectory_monitoring.benchmark_distilled_ood import (
    adapter_metadata,
    validate_config,
    validate_inputs,
    validate_jobs,
)
from gleipnir.monitoring_systems_screen import (
    atomic_write_json,
    atomic_write_jsonl,
    read_jsonl,
    sha256_file,
)
from gleipnir.qwen35_fast_training import ensure_qwen35_long_trajectory_kernels

ROOT = Path("results/monitoring_mil_ood")
JOB = "mil-pct020-lr2em05-epochs3-seed0"


def build_config(template: dict, jobs_path: Path) -> dict:
    """Preserve the historical data/prompt contract and select only MIL3."""
    config = json.loads(json.dumps(template))
    config.update(
        campaign_id="monitoring-mil3-frozen-ood-v1",
        hypothesis="The fixed ID-selected monitoring-only MIL3 transfers to OOD.",
        expected_model_groups=["4b"],
    )
    group = config["model_groups"]["4b"]
    group.update(
        jobs=str(jobs_path),
        jobs_sha256=sha256_file(jobs_path),
        expected_jobs=[JOB],
        parity_job=JOB,
        expected_target="kimi_soft_plus_auxiliary",
        evaluate_base=False,
    )
    config["model_groups"] = {"4b": group}
    config["engine"]["gdn_prefill_backend"] = "flashinfer"
    return config


def main() -> None:
    from experiments.monitoring_lr_sweep.run_lambda import gpu_training_environment
    from experiments.monitoring_objective_ablation.run_lambda import run_serving_parity
    from experiments.monitoring_subset_duration.run import validate_completed

    ROOT.mkdir(parents=True, exist_ok=False)
    status = {"state": "running", "phase": "preflight", "job": JOB}
    atomic_write_json(ROOT / "status.json", status)
    try:
        hardware = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).splitlines()
        if len(hardware) != 2 or any(int(r.rsplit(",", 1)[1]) > 1024 for r in hardware):
            raise RuntimeError("two idle GPUs required")
        atomic_write_json(ROOT / "hardware.json", {"gpus": hardware})
        jobs = read_jsonl(Path("results/monitoring_subset_duration/jobs.jsonl"))
        selected = [j for j in jobs if j["job_name"] == JOB]
        if len(selected) != 1:
            raise ValueError("selected checkpoint missing or duplicated")
        validate_completed(selected[0])
        atomic_write_json(
            ROOT / "adapter_provenance.json", adapter_metadata(selected[0])
        )
        atomic_write_jsonl(ROOT / "jobs.jsonl", selected)
        template = json.loads(
            Path(
                "experiments/tool_trajectory_monitoring/distillation_ood_benchmark.json"
            ).read_text()
        )
        config = build_config(template, ROOT / "jobs.jsonl")
        config["scope"]["manifest_sha256"] = sha256_file(
            Path(config["scope"]["manifest"])
        )
        validate_config(config)
        validate_inputs(config)
        validate_jobs(config, "4b")
        config_path = ROOT / "ood_benchmark.json"
        atomic_write_json(config_path, config)
        serving = runtime_environment("0")
        os.environ.update(serving)
        fast = gpu_training_environment(ensure_qwen35_long_trajectory_kernels(), 0)
        run_serving_parity(config_path, ROOT, fast, serving)
        subprocess.run(
            [
                sys.executable,
                "-u",
                "-m",
                "experiments.monitoring_prefix_supervision.resume_evaluation",
                "--root",
                str(ROOT),
                "--job",
                JOB,
                "--config",
                str(config_path),
                "--evaluation-dir",
                "ood_evaluation",
                "--fresh",
                "--skip-id-summary",
            ],
            check=True,
            env=serving,
        )
    except BaseException as error:
        status.update(state="failed", error=repr(error))
        atomic_write_json(ROOT / "status.json", status)
        raise


if __name__ == "__main__":
    main()
