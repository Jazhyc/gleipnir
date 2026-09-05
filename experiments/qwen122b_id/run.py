"""Prepare and supervise the fixed two-GPU local-teacher ID benchmark."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from experiments.adapter_capacity_scaling.run_lambda import runtime_environment
from experiments.tool_trajectory_monitoring.benchmark_qwen_ood import (
    render_margin_prompt,
    validate_config,
    validate_inputs,
)
from gleipnir.monitoring_systems_screen import atomic_write_json, sha256_file

ROOT = Path("results/qwen122b_id")


def build_config(settings: dict[str, Any], lengths: list[int]) -> dict[str, Any]:
    """Reuse the scoring contract, replacing all study-specific OOD metadata."""
    if len(lengths) != 3012 or max(lengths) >= 32768:
        raise ValueError("ID token audit is incomplete or exceeds context")
    config = json.loads(Path(settings["template"]).read_text())
    config.update(
        campaign_id=settings["campaign_id"],
        hypothesis="Local FP8 teacher quality and throughput on fixed ID rows",
        intervention=(
            f"Official {settings['model']['id']}, TP2, full teacher instruction"
        ),
        model=settings["model"],
        scope=settings["scope"],
        engine={
            **settings["engine"],
            "audited_max_prompt_tokens": max(lengths),
            "audited_total_prompt_tokens": sum(lengths),
        },
        baselines=[
            "Existing full-teacher ID scores and compact-prompt Gleipnir ID "
            "scores are contextual, not matched precision/prompt ablations"
        ],
        stop_conditions=[
            "See experiments/qwen122b_id/README.md; fail closed on input, "
            "tokenizer, canary, memory, or progress failure"
        ],
    )
    validate_config(config)
    return config


def prepare(config_name: str = "config") -> None:
    from transformers import AutoTokenizer

    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).parent.resolve())
    ):
        settings = OmegaConf.to_container(
            compose(config_name=config_name), resolve=True
        )
    root = Path(settings["result_dir"])
    if (root / "manifest.json").exists():
        raise ValueError("reuse the frozen prepared contract")
    provisional = build_config(settings, [1] * 3012)
    rows = validate_inputs(provisional)
    tokenizer = AutoTokenizer.from_pretrained(
        settings["model"]["id"], revision=settings["model"]["revision"]
    )
    prompt = provisional["prompt"]
    lengths = [
        len(
            tokenizer.encode(
                render_margin_prompt(
                    tokenizer,
                    row["prompt"],
                    enable_thinking=False,
                    assistant_suffix=prompt["assistant_suffix"],
                    decision_prefix=prompt["decision_prefix"],
                ),
                add_special_tokens=False,
            )
        )
        for row in rows
    ]
    config = build_config(settings, lengths)
    atomic_write_json(root / "benchmark.json", config)
    atomic_write_json(
        root / "manifest.json",
        {
            "files": {
                str(path): sha256_file(path)
                for path in (
                    root / "benchmark.json",
                    Path(config["scope"]["input"]),
                    Path(config["scope"]["manifest"]),
                )
            },
            "authoring_sha256": sha256_file(
                Path(__file__).with_name(f"{config_name}.yaml")
            ),
            "tokenizer_revision": settings["model"]["revision"],
        },
    )
    print(json.dumps(config["engine"]), flush=True)


def gpu_memory() -> list[int]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    return [int(line) for line in output.splitlines()]


def run(root: Path) -> None:
    for name, digest in json.loads((root / "manifest.json").read_text())[
        "files"
    ].items():
        if sha256_file(Path(name)) != digest:
            raise ValueError(f"frozen contract drift: {name}")
    memory = gpu_memory()
    if len(memory) != 2 or max(memory) > 1024:
        raise RuntimeError(f"both GPUs must be idle before loading: {memory}")
    logs = Path("logs/lambda") / root.name
    logs.mkdir(parents=True, exist_ok=True)
    status: dict[str, Any] = {
        "state": "running",
        "phase": "initializing",
        "started_at_unix": time.time(),
        "revision": os.environ.get("GLEIPNIR_COMMIT"),
        "gpu_memory_peak_observed_mib": memory,
    }

    def update(**fields: Any) -> None:
        status.update(fields, updated_at_unix=time.time())
        atomic_write_json(root / "status.json", status)

    child = None
    try:
        update()
        with (logs / "evaluation.log").open("a") as stream:
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "experiments.tool_trajectory_monitoring.benchmark_qwen_ood",
                    "--config",
                    str(root / "benchmark.json"),
                    "--output",
                    str(root / "evaluation"),
                ],
                env=runtime_environment("0,1"),
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            last_progress = time.time()
            previous_count = -1
            while child.poll() is None:
                predictions = root / "evaluation/predictions.jsonl"
                count = (
                    sum(1 for _ in predictions.open()) if predictions.exists() else 0
                )
                if count != previous_count:
                    last_progress, previous_count = time.time(), count
                memory = gpu_memory()
                peak = [
                    max(a, b)
                    for a, b in zip(
                        status["gpu_memory_peak_observed_mib"], memory, strict=True
                    )
                ]
                ready = (root / "evaluation/canary_result.json").exists()
                update(
                    phase="evaluation" if ready else "initializing_and_canary",
                    predictions=count,
                    pid=child.pid,
                    gpu_memory_peak_observed_mib=peak,
                )
                print("PROCESS_WATCHDOG", json.dumps(status), flush=True)
                if time.time() - last_progress > (1200 if ready else 1800):
                    raise RuntimeError("evaluation stalled without prediction progress")
                try:
                    child.wait(timeout=600 if ready else 30)
                except subprocess.TimeoutExpired:
                    pass
            if child.returncode:
                raise RuntimeError(f"evaluator exited {child.returncode}")
        result = json.loads((root / "evaluation/result.json").read_text())
        if result["rows"] != 3012:
            raise ValueError("incomplete final ID result")
        update(state="complete", phase="complete", completed_at_unix=time.time())
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        update(state="failed", error=repr(error))
        raise


def dependency_ready(dependency: Path) -> bool:
    """Never start a dependent GPU run on a failed or incomplete predecessor."""
    state = json.loads((dependency / "status.json").read_text())["state"]
    if state == "failed":
        raise RuntimeError(f"predecessor failed: {dependency}")
    if state != "complete":
        return False
    result = json.loads((dependency / "evaluation/result.json").read_text())
    if result["rows"] != 3012:
        raise ValueError("predecessor has an incomplete evaluation")
    return True


def wait_for_dependency(root: Path, dependency: Path) -> None:
    try:
        while not dependency_ready(dependency):
            atomic_write_json(
                root / "status.json",
                {
                    "state": "queued",
                    "phase": "waiting_for_predecessor",
                    "predecessor": str(dependency),
                    "updated_at_unix": time.time(),
                },
            )
            print(f"QUEUE_WAIT for {dependency}", flush=True)
            time.sleep(600)
        for _ in range(12):
            if max(gpu_memory()) <= 1024:
                return
            time.sleep(5)
        raise RuntimeError("predecessor completed but GPUs remain occupied")
    except BaseException as error:
        atomic_write_json(
            root / "status.json",
            {
                "state": "failed",
                "phase": "dependency_gate",
                "error": repr(error),
                "updated_at_unix": time.time(),
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--result-dir", type=Path, default=ROOT)
    parser.add_argument("--config-name", default="config")
    parser.add_argument("--after-result-dir", type=Path)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.config_name)
    else:
        if args.after_result_dir:
            wait_for_dependency(args.result_dir, args.after_result_dir)
        run(args.result_dir)


if __name__ == "__main__":
    main()
