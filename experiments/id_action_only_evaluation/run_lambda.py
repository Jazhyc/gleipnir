"""Parity-check cleaned ID inputs, then use two independent vLLM workers."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from experiments.adapter_capacity_scaling.run_lambda import runtime_environment
from experiments.id_action_only_evaluation.prepare import CONFIG, JOB, ROOT
from experiments.monitoring_lr_sweep.run_lambda import gpu_training_environment
from experiments.tool_trajectory_monitoring.benchmark_distilled_ood import (
    adapter_metadata,
    validate_config,
    validate_inputs,
    validate_jobs,
)
from experiments.tool_trajectory_monitoring.prepare_distillation_ood import (
    atomic_write_json,
)
from gleipnir.qwen35_fast_training import ensure_qwen35_long_trajectory_kernels

LOGS = Path("logs/lambda/id_action_only_evaluation")


def main(
    *, config_path: Path = CONFIG, output_root: Path = ROOT, log_root: Path = LOGS
) -> None:
    log_root.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text())
    validate_config(config)
    validate_inputs(config)
    job = validate_jobs(config, "4b")[0]
    details = adapter_metadata(job)
    for key in ("source_sha256", "destination_sha256"):
        if details["rebase_manifest"][key] != config["adapter_checksums"][key]:
            raise ValueError("Adapter differs from historical standard Gleipnir 4B")
    hardware = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
            "--format=csv",
        ],
        text=True,
    )
    processes = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    if processes or len(hardware.strip().splitlines()) != 3:
        raise RuntimeError("Two idle GPUs required")
    atomic_write_json(
        output_root / "hardware.json",
        {
            "gpu_query": hardware,
            "compute_processes": processes,
            "time": time.time(),
        },
    )
    status = {
        "state": "running",
        "phase": "kernel_preflight",
        "started_at_unix": time.time(),
    }

    def update(phase: str) -> None:
        status.update(phase=phase, updated_at_unix=time.time())
        atomic_write_json(output_root / "status.json", status)

    def run(module: str, args: list[str], env: dict[str, str], log: str) -> None:
        with (log_root / log).open("a") as handle:
            subprocess.run(
                [sys.executable, "-u", "-m", module, *args],
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )

    try:
        update("kernel_preflight")
        fast = ensure_qwen35_long_trajectory_kernels()
        common = ["--config", str(config_path), "--model-size", "4b", "--output-root"]
        parity = output_root / "parity"
        update("cleaned_input_parity")
        with ThreadPoolExecutor(max_workers=2) as pool:
            eager = pool.submit(
                run,
                "experiments.tool_trajectory_monitoring.evaluate_distilled_ood_causal",
                [*common, str(parity / "eager")],
                gpu_training_environment(fast, 0),
                "parity_eager_gpu0.log",
            )
            serving = pool.submit(
                run,
                "experiments.tool_trajectory_monitoring.benchmark_distilled_ood",
                [
                    *common,
                    str(parity / "vllm"),
                    "--only-job",
                    JOB,
                    "--include-base",
                    "--canary-only",
                ],
                runtime_environment("1"),
                "parity_vllm_gpu1.log",
            )
            eager.result()
            serving.result()
        run(
            "experiments.tool_trajectory_monitoring.compare_distilled_ood_parity",
            [
                "--eager-root",
                str(parity / "eager"),
                "--vllm-root",
                str(parity / "vllm"),
                "--model-size",
                "4b",
                "--job-name",
                JOB,
                "--output",
                str(parity / "4b.json"),
            ],
            runtime_environment("0"),
            "parity_comparison.log",
        )
        update("id_evaluation_sharded")
        run(
            "experiments.monitoring_prefix_supervision.resume_evaluation",
            [
                "--root",
                str(output_root),
                "--job",
                JOB,
                "--config",
                str(config_path),
                "--fresh",
                "--skip-id-summary",
            ],
            runtime_environment("0,1"),
            "sharded_launcher.log",
        )
    except BaseException as error:
        status.update(state="failed", error=repr(error))
        update("failed")
        raise


if __name__ == "__main__":
    main()
