"""Detached final-head precision test with frozen baselines and owned cleanup."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
import shutil
import subprocess
import time

import numpy as np
import yaml

from experiments.teacher_disagreement_filtering.run import retire
from gleipnir.campaigns.monitoring.contract import Campaign
from gleipnir.campaigns.monitoring.evaluation import (
    attach_inputs,
    id_metrics,
    optimized,
)
from gleipnir.data.monitoring import file_hash, read_rows, write_json
from gleipnir.serving.monitor_score import score_payload, validate_score_response

from .prepare import DATA, NAME, OOD, OUT, ROOT


class HeadContext(Campaign):
    @property
    def adapter(self):
        return self.root / self.config["parent_campaign"] / "4b/monitor"

    @property
    def splits(self):
        return ("id", "ood")

    def check(self):
        manifest = json.loads((self.data / "manifest.json").read_text())
        for section in ("files_sha256", "source_sha256"):
            for path, expected in manifest[section].items():
                if file_hash(self.root / path) != expected:
                    raise ValueError(f"FP32 head input/source drift: {path}")
        return manifest


def context():
    path = DATA / "config.yaml"
    config = yaml.safe_load(path.read_text())
    if (
        config["campaign_id"] != NAME
        or config["evaluation"]["serving_precision"] != "bf16"
        or config["evaluation"]["head_compute_precision"] != "fp32"
    ):
        raise ValueError("head precision contract changed")
    return HeadContext(ROOT, path, config)


def reuse_merge(ctx):
    artifact = json.loads((ROOT / OOD / "merged_artifact.json").read_text())
    from pathlib import Path

    slot = Path(ctx.config["merged_model"])
    if slot.is_symlink() or not slot.exists():
        raise ValueError("strict shared merge missing or unexpectedly linked")
    for name, expected in artifact["files_sha256"].items():
        if file_hash(slot / name) != expected:
            raise ValueError(f"existing strict merge changed: {name}")
    for name in ("merged_artifact.json", "merged_parity.json", "canary_workload.jsonl"):
        shutil.copyfile(ROOT / OOD / name, OUT / name)
    parity = json.loads((OUT / "merged_parity.json").read_text())
    if not parity["passed"]:
        raise ValueError("original strict merge parity failed")
    write_json(
        OUT / "merge_reuse.json",
        {
            "passed": True,
            "source": OOD,
            "policy": "exact existing strict merge, original FP32-master reference",
            "artifact_sha256": file_hash(OUT / "merged_artifact.json"),
        },
    )


def install_head_override():
    from gleipnir.serving import bf16_monitor

    original = bf16_monitor.command

    def command(parent, sources):
        result = original(parent, sources)
        result[result.index("--worker-cls") + 1] = (
            "experiments.monitor_score_resolution.worker.Fp32HeadWorker"
        )
        index = result.index("--additional-config") + 1
        additional = json.loads(result[index])
        additional["monitor_head_precision"] = "fp32"
        result[index] = json.dumps(additional, sort_keys=True)
        return result

    bf16_monitor.command = command


def summarize(ctx):
    native = json.loads((OUT / "bf16_audit.json").read_text())
    if (
        not native["head_computation"]["passed"]
        or native["head_computation"]["output_dtype"] != "torch.float32"
    ):
        raise ValueError("actual FP32 head dispatch missing")
    complete = json.loads((OUT / "evaluation/complete.json").read_text())
    if not complete["parity_passed"] or complete["rows"] != 9407:
        raise ValueError("candidate coverage/parity failed")
    result = {"status": "complete", "promotion": False, "cases": {}}
    for split, expected_count in (("id", 3012), ("ood", 6395)):
        inputs = read_rows(ctx.input(split))
        path = OUT / "evaluation" / f"{split}.jsonl"
        rows = read_rows(path)
        if (
            len(rows) != expected_count
            or file_hash(path) != complete["files_sha256"][split]
        ):
            raise ValueError("prediction count/hash drift")
        for row in rows:
            validate_score_response(row, row["prompt_tokens"])
        candidate = id_metrics(attach_inputs(rows, inputs))
        parent = ROOT / ctx.config["matched_baselines"][split]
        baseline_rows = read_rows(parent / "evaluation" / f"{split}.jsonl")
        baseline = id_metrics(attach_inputs(baseline_rows, inputs))
        for row, old in zip(rows, baseline_rows, strict=True):
            if (
                row["prompt_sha256"] != old["prompt_sha256"]
                or row["prompt_tokens"] != old["prompt_tokens"]
            ):
                raise ValueError("candidate/baseline rendered prompt drift")
        # Same-pass counterfactual: round only the FP32 head outputs to BF16.
        # Not a second inference run and not a replacement for the saved control.
        import torch

        rounded = []
        for row in rows:
            logits = (
                torch.tensor(row["logits"], dtype=torch.float32)
                .to(torch.bfloat16)
                .float()
                .tolist()
            )
            rounded.append({**row, **score_payload(logits, row["prompt_tokens"])})
        rounded_metrics = id_metrics(rounded)
        current = np.array([r["score"] for r in rows])
        old = np.array([r["score"] for r in baseline_rows])
        timing = json.loads((OUT / "evaluation" / f"{split}_timing.json").read_text())
        baseline_timing = json.loads(
            (parent / "evaluation" / f"{split}_timing.json").read_text()
        )
        main, before = (
            candidate["metrics"]["macro"]["macro"],
            baseline["metrics"]["macro"]["macro"],
        )
        result["cases"][split] = {
            "candidate": candidate,
            "cached_bf16": baseline,
            "same_pass_head_output_rounded_bf16": rounded_metrics,
            "candidate_minus_cached_bf16": {k: main[k] - before[k] for k in main},
            "unique_scores": len(set(current)),
            "baseline_unique_scores": len(set(old)),
            "mean_absolute_score_difference": float(np.abs(current - old).mean()),
            "threshold_flips": int(np.sum((current >= 0.5) != (old >= 0.5))),
            "input_tokens_per_second": timing["tokens"] / timing["seconds"],
            "baseline_input_tokens_per_second": baseline_timing["tokens"]
            / baseline_timing["seconds"],
            "requests_per_second": len(rows) / timing["seconds"],
            "latency_seconds_median": float(
                np.median([r["latency_seconds"] for r in rows])
            ),
            "seconds": timing["seconds"],
            "predictions_sha256": file_hash(path),
        }
        print(
            "fp32_head_result",
            split,
            result["cases"][split]["candidate_minus_cached_bf16"],
            flush=True,
        )
    write_json(OUT / "comparison.json", result)
    return result


def run():
    ctx = context()
    ctx.check()
    OUT.mkdir(parents=True, exist_ok=True)
    status = OUT / "status.json"
    if status.exists():
        raise ValueError("preserve existing head-run receipt")
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).strip()
    if gpu != ctx.config["gpu_uuid"]:
        raise ValueError("authorized GPU identity drift")
    state = {
        "status": "running",
        "driver_pid": os.getpid(),
        "started_at_unix": time.time(),
    }
    write_json(status, state)
    try:
        reuse_merge(ctx)
        state["stage"] = "score"
        write_json(status, state)
        command = [
            ctx.config["serving_python"],
            "-m",
            "experiments.b200_vllm031.runtime",
            "-m",
            "experiments.monitor_score_resolution.run",
            "--stage",
            "score",
        ]
        ctx.logs.mkdir(parents=True, exist_ok=True)
        with (ctx.logs / "score.log").open("x") as log:
            subprocess.run(
                command,
                cwd=ROOT,
                check=True,
                timeout=3600,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={
                    **os.environ,
                    "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}",
                    "GLEIPNIR_CAMPAIGN_DRIVER_PID": str(os.getpid()),
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                },
            )
        summarize(ctx)
        retire(ctx)
    except BaseException as error:
        if (ctx.serving / "server.json").exists():
            try:
                retire(ctx)
            except Exception as cleanup:
                state["cleanup_error"] = str(cleanup)
        state.update(status="failed", error=str(error), finished_at_unix=time.time())
        write_json(status, state)
        raise
    state.update(
        status="complete",
        finished_at_unix=time.time(),
        comparison_sha256=file_hash(OUT / "comparison.json"),
    )
    write_json(status, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["score"])
    args = parser.parse_args()
    if args.stage:
        install_head_override()
        asyncio.run(optimized(context()))
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        with (OUT / "runner.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run()
