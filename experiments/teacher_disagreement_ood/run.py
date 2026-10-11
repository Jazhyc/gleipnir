"""Evaluate the fixed strict-filtered adapter once against cached OOD scores."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import yaml

from experiments.teacher_disagreement_filtering.run import release_merge, retire
from experiments.teacher_disagreement_ood.prepare import (
    DATA,
    NAME,
    OUT,
    PARENT,
    ROOT,
    bind_cached,
)
from gleipnir.campaigns.monitoring.contract import Campaign
from gleipnir.campaigns.monitoring.evaluation import (
    attach_inputs,
    id_metrics,
    optimized,
)
from gleipnir.campaigns.monitoring.training import merge
from gleipnir.campaigns.runtime import training_environment
from gleipnir.data.monitoring import file_hash, read_rows, write_json
from gleipnir.serving.monitor_score import validate_score_response


class OODContext(Campaign):
    """Reuse validated serving helpers while retaining the original adapter files."""

    @property
    def adapter(self) -> Path:
        return self.root / self.config["parent_campaign"] / "4b/monitor"

    @property
    def splits(self) -> tuple[str, ...]:
        return ("ood",)

    def check(self) -> dict:
        manifest = json.loads((self.data / "manifest.json").read_text())
        for section in ["files_sha256", "source_sha256"]:
            for path, expected in manifest[section].items():
                if file_hash(self.root / path) != expected:
                    raise ValueError(f"OOD follow-up input/source drift: {path}")
        return manifest


def context() -> OODContext:
    path = DATA / "config.yaml"
    config = yaml.safe_load(path.read_text())
    if (
        config["campaign_id"] != NAME
        or config["evaluations"] != ["ood"]
        or config["evaluation"]["serving_precision"] != "bf16"
        or config["evaluation"]["populations"] != {"ood": 6395}
    ):
        raise ValueError("fixed candidate/precision/population contract drift")
    return OODContext(ROOT, path, config)


def reconstruct(ctx: OODContext) -> None:
    """Replace the last ID merge and reuse parity only for identical new weights."""
    ctx.check()
    slot = Path(ctx.config["merged_model"])
    if slot.exists():
        last = Campaign.load(
            ROOT, ROOT / "data/teacher-disagreement-delta020-sdpa04/config.yaml"
        )
        release_merge(last)
    merge(ctx)
    original = json.loads((ROOT / PARENT / "merged_artifact.json").read_text())
    actual = json.loads((OUT / "merged_artifact.json").read_text())
    for key in [
        "files_sha256",
        "source_files_sha256",
        "adapter_sha256",
        "adapter_config_sha256",
    ]:
        if original[key] != actual[key]:
            raise ValueError(f"reconstructed merge differs: {key}")
    receipt = json.loads((ROOT / PARENT / "merged_parity.json").read_text())
    if not receipt["passed"]:
        raise ValueError("original strict-filtered merge parity failed")
    receipt.update(
        performed_this_run=False,
        policy="reuse_identical_merge_and_original_fp32_reference",
        source_receipt_sha256=file_hash(ROOT / PARENT / "merged_parity.json"),
        merged_artifact_sha256=file_hash(OUT / "merged_artifact.json"),
    )
    write_json(OUT / "merged_parity.json", receipt)
    shutil.copyfile(
        ROOT / PARENT / "canary_workload.jsonl", OUT / "canary_workload.jsonl"
    )
    print("ood_identical_merge_and_reference_verified", flush=True)


def summarize(ctx: OODContext) -> dict:
    """Verify full raw outputs and compare fixed models on identical membership."""
    ctx.check()
    canonical = read_rows(ctx.input("ood"))
    predictions = OUT / "evaluation/ood.jsonl"
    done = json.loads((OUT / "evaluation/complete.json").read_text())
    if (
        not done["parity_passed"]
        or done["rows"] != 6395
        or done["files_sha256"]["ood"] != file_hash(predictions)
    ):
        raise ValueError("OOD evaluation completion/hash drift")
    rows = read_rows(predictions)
    for row in rows:
        validate_score_response(row, row["prompt_tokens"])
    candidate = id_metrics(attach_inputs(rows, canonical))
    references = {
        name: id_metrics(bind_cached(read_rows(ROOT / path), canonical))
        for name, path in ctx.config["cached_references"].items()
    }
    main = candidate["metrics"]["macro"]["macro"]
    differences = {
        name: {
            key: main[key] - metrics["metrics"]["macro"]["macro"][key]
            for key in ["pauroc_at_20", "auroc", "brier"]
        }
        for name, metrics in references.items()
    }
    report = {
        "status": "complete",
        "candidate": "strict_delta_0",
        "selected_before_ood": True,
        "candidate_metrics": candidate,
        "cached_references": references,
        "candidate_minus_reference": differences,
        "primary_reference": "gleipnir4b",
        "promotion": False,
        "qualification": (
            "Cached Gleipnir 4B uses mixed training data; cached references use "
            "historical dynamic-LoRA serving. No matched causal attribution."
        ),
        "rows": len(rows),
        "predictions_sha256": file_hash(predictions),
        "manifest_sha256": file_hash(DATA / "manifest.json"),
        "all_scores_membership_and_metrics_verified": True,
    }
    write_json(OUT / "ood_summary.json", report)
    # The shared helper owns server/parity checks; OOD metrics are supplied here.
    serving_summary = json.loads((OUT / "summary.json").read_text())
    serving_summary["evaluations"]["ood"] = candidate
    write_json(OUT / "summary.json", serving_summary)
    print("ood_verified_summary", main, differences, flush=True)
    return report


def run() -> None:
    ctx = context()
    ctx.check()
    OUT.mkdir(parents=True, exist_ok=True)
    status = OUT / "status.json"
    if status.exists():
        raise ValueError("OOD follow-up already launched; preserve its receipts")
    actual_gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).strip()
    if actual_gpu != ctx.config["gpu_uuid"]:
        raise ValueError("authorized GPU identity changed")
    state = {
        "status": "running",
        "driver_pid": os.getpid(),
        "started_at_unix": time.time(),
    }
    write_json(status, state)
    try:
        for stage in ["merge", "score"]:
            python = ctx.config[
                "training_python" if stage == "merge" else "serving_python"
            ]
            command = [
                python,
                "-m",
                "experiments.teacher_disagreement_ood.run",
                "--stage",
                stage,
            ]
            if stage == "score":
                command[1:1] = ["-m", "experiments.b200_vllm031.runtime"]
            env = (
                training_environment(ROOT, NAME)
                if stage == "merge"
                else dict(os.environ)
            )
            env.update(
                PYTHONPATH=f"{ROOT / 'src'}:{ROOT}:{env.get('PYTHONPATH', '')}",
                GLEIPNIR_CAMPAIGN_DRIVER_PID=str(os.getpid()),
                HF_HUB_OFFLINE="1",
                TRANSFORMERS_OFFLINE="1",
            )
            ctx.logs.mkdir(parents=True, exist_ok=True)
            with (ctx.logs / f"{stage}.log").open("x") as log:
                state.update(stage=stage)
                write_json(status, state)
                subprocess.run(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=3600,
                )
        summarize(ctx)
        retire(ctx)
    except BaseException as error:
        if (ctx.serving / "server.json").exists():
            try:
                retire(ctx)
            except Exception as cleanup_error:
                state["cleanup_error"] = str(cleanup_error)
        state.update(status="failed", error=str(error), failed_at_unix=time.time())
        write_json(status, state)
        raise
    state.update(
        status="complete",
        finished_at_unix=time.time(),
        summary_sha256=file_hash(OUT / "ood_summary.json"),
    )
    write_json(status, state)
    print("strict_filter_ood_complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["merge", "score"])
    args = parser.parse_args()
    if args.stage == "merge":
        reconstruct(context())
    elif args.stage == "score":
        asyncio.run(optimized(context()))
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        with (OUT / "runner.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run()
