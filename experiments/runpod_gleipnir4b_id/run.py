"""Prepare, parity-gate, and evaluate the released 4B checkpoint on Runpod."""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from experiments.tool_trajectory_monitoring.benchmark_distilled_ood import (
    adapter_metadata,
    sha256_file,
    validate_config,
    validate_inputs,
    validate_jobs,
)
from experiments.tool_trajectory_monitoring.compare_distilled_ood_parity import (
    compare_predictions,
)
from experiments.tool_trajectory_monitoring.qwen_reasoning_core import (
    atomic_write_json,
    atomic_write_jsonl,
    load_json,
    load_jsonl,
)
from gleipnir.calibration import binary_calibration
from gleipnir.qwen35_fast_training import ensure_qwen35_long_trajectory_kernels

CONFIG = Path("experiments/runpod_gleipnir4b_id/config.json")
OUTPUT = Path("results/runpod_gleipnir4b_id")
LOGS = Path("logs/runpod/runpod_gleipnir4b_id")
JOB = "soft-n21837-mixed-qwen35-4b-seed0"


def require_flashinfer_prefill(log: Path) -> None:
    """Reject a requested FlashInfer backend that silently selected a fallback."""
    if "Using FlashInfer GDN prefill kernel" not in log.read_text():
        raise RuntimeError("vLLM did not activate the required FlashInfer GDN prefill")


def prepare() -> None:
    """Relocate only the frozen job paths, preserving scientific provenance."""
    config = load_json(CONFIG)
    release = load_json(Path("model_releases/hf/release_config.json"))["models"]["4b"]
    job = load_jsonl(Path("results/id_cot_only_evaluation/jobs.jsonl"))[0]
    if job["job_name"] != JOB:
        raise ValueError("Historical release job identity drifted")
    for kind, directory_key in (("causal", "source_dir"), ("serving", "serving_dir")):
        destination = OUTPUT / "artifacts" / kind
        destination.mkdir(parents=True, exist_ok=True)
        files = [
            "adapter_config.json",
            "adapter_model.safetensors",
            "training_metadata.json",
        ]
        if kind == "serving":
            files.append("rebase_manifest.json")
        for name in files:
            temporary = destination / f"{name}.tmp"
            shutil.copy2(Path(release[directory_key]) / name, temporary)
            temporary.replace(destination / name)
    for key, value in (
        ("causal_adapter_dir", str(OUTPUT / "artifacts/causal")),
        ("model_dir", str(OUTPUT / "artifacts/serving")),
        ("output_dir", str(OUTPUT / "artifacts")),
    ):
        job[key] = value
    atomic_write_jsonl(OUTPUT / "jobs.jsonl", [job])
    validate_config(config)
    validate_inputs(config)
    details = adapter_metadata(validate_jobs(config, "4b")[0])
    for key, expected in config["adapter_checksums"].items():
        if details["rebase_manifest"][key] != expected:
            raise ValueError(f"Released adapter {key} drifted")
    atomic_write_json(
        OUTPUT / "preparation.json",
        {
            "config_sha256": sha256_file(CONFIG),
            "jobs_sha256": sha256_file(OUTPUT / "jobs.jsonl"),
            "input_sha256": config["scope"]["input_sha256"],
            "manifest_sha256": config["scope"]["manifest_sha256"],
            "adapter_checksums": config["adapter_checksums"],
        },
    )


def summarize() -> dict[str, Any]:
    """Audit exact coverage and add calibration to the shared evaluator output."""
    config = load_json(CONFIG)
    inputs = validate_inputs(config)
    directory = OUTPUT / "evaluation/4b/adapters" / JOB
    predictions = load_jsonl(directory / "predictions.jsonl")
    expected = {str(row["id"]): row for row in inputs}
    observed = [str(row["id"]) for row in predictions]
    if len(observed) != len(set(observed)) or set(observed) != set(expected):
        raise ValueError("ID prediction coverage is incomplete or duplicated")
    for row in predictions:
        original = expected[str(row["id"])]
        if (
            row["source"] != original["metadata"]["source_dataset"]
            or int(row["label"]) != original["metadata"]["ground_truth"]
            or row["config_sha256"] != sha256_file(CONFIG)
        ):
            raise ValueError("Prediction identity or execution contract drifted")
        if not math.isfinite(float(row["score"])):
            raise ValueError("Prediction contains a non-finite score")
    views = {"pooled": predictions}
    views.update(
        {
            source: [r for r in predictions if r["source"] == source]
            for source in config["scope"]["sources"]
        }
    )
    result = load_json(directory / "result.json")
    result["calibration"] = {
        name: {
            str(n): binary_calibration(
                [r["label"] for r in rows], [r["score"] for r in rows], n
            )
            for n in (5, 10, 20)
        }
        for name, rows in views.items()
    }
    result["artifacts"] = {
        "predictions_sha256": sha256_file(directory / "predictions.jsonl"),
        "result_sha256": sha256_file(directory / "result.json"),
        "preparation": load_json(OUTPUT / "preparation.json"),
    }
    baseline = (
        Path("results/id_cot_only_evaluation/id_evaluation/4b/adapters")
        / JOB
        / "predictions.jsonl"
    )
    if baseline.is_file():
        result["historical_score_agreement"] = compare_predictions(
            baseline, directory / "predictions.jsonl"
        )
    atomic_write_json(OUTPUT / "summary.json", result)
    return result


def run() -> None:
    config = load_json(CONFIG)
    preparation = load_json(OUTPUT / "preparation.json")
    if preparation["config_sha256"] != sha256_file(CONFIG) or preparation[
        "jobs_sha256"
    ] != sha256_file(OUTPUT / "jobs.jsonl"):
        raise ValueError("Prepared execution contract drifted")
    validate_inputs(config)
    adapter_metadata(validate_jobs(config, "4b")[0])
    LOGS.mkdir(parents=True, exist_ok=True)
    status: dict[str, Any] = {
        "state": "running",
        "started_at_unix": time.time(),
        "config_sha256": sha256_file(CONFIG),
    }

    def phase(name: str) -> None:
        status.update(phase=name, updated_at_unix=time.time())
        atomic_write_json(OUTPUT / "status.json", status)
        print(f"phase={name}", flush=True)

    def call(
        module: str, arguments: list[str], environment: dict[str, str], log: str
    ) -> None:
        with (LOGS / log).open("a") as handle:
            subprocess.run(
                [sys.executable, "-u", "-m", module, *arguments],
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=3600,
            )

    try:
        phase("kernel_preflight")
        os.environ["FLA_DISABLE_BACKEND_DISPATCH"] = "1"
        kernels = Path(".cache/kernels").resolve()
        environment = ensure_qwen35_long_trajectory_kernels(
            kernels / "fla",
            kernels / "causal_conv1d",
            kernels / "triton",
        )
        environment["CUDA_VISIBLE_DEVICES"] = "0"
        with (LOGS / "kernel_canary.log").open("a") as handle:
            subprocess.run(
                [sys.executable, "-u", "scripts/qwen35_fast_kernel_canary.py"],
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=600,
            )
        common = ["--config", str(CONFIG), "--model-size", "4b", "--output-root"]
        phase("master_parity")
        call(
            "experiments.tool_trajectory_monitoring.evaluate_distilled_ood_causal",
            [*common, str(OUTPUT / "parity/eager")],
            environment,
            "parity_eager.log",
        )
        serving = dict(os.environ)
        serving["CUDA_VISIBLE_DEVICES"] = "0"
        phase("serving_parity")
        call(
            "experiments.tool_trajectory_monitoring.benchmark_distilled_ood",
            [
                *common,
                str(OUTPUT / "parity/vllm"),
                "--only-job",
                JOB,
                "--include-base",
                "--canary-only",
                "--no-watchdog",
            ],
            serving,
            "parity_vllm.log",
        )
        require_flashinfer_prefill(LOGS / "parity_vllm.log")
        call(
            "experiments.tool_trajectory_monitoring.compare_distilled_ood_parity",
            [
                "--eager-root",
                str(OUTPUT / "parity/eager"),
                "--vllm-root",
                str(OUTPUT / "parity/vllm"),
                "--model-size",
                "4b",
                "--job-name",
                JOB,
                "--output",
                str(OUTPUT / "parity/report.json"),
            ],
            serving,
            "parity_comparison.log",
        )
        phase("id_evaluation")
        call(
            "experiments.tool_trajectory_monitoring.benchmark_distilled_ood",
            [*common, str(OUTPUT / "evaluation"), "--only-job", JOB],
            serving,
            "evaluation.log",
        )
        require_flashinfer_prefill(LOGS / "evaluation.log")
        summarize()
        status.update(
            state="complete", elapsed_seconds=time.time() - status["started_at_unix"]
        )
        phase("complete")
    except BaseException as error:
        status.update(state="failed", error=repr(error))
        phase("failed")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run", "summarize"])
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "summarize": summarize}[args.action]()


if __name__ == "__main__":
    main()
