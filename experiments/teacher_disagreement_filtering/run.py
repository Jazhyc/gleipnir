"""Detached, bounded serial execution and automatic ID sweep summary."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil

from experiments.teacher_disagreement_filtering.prepare import (
    OUTPUT,
    ROOT,
    verify_preparation,
)
from gleipnir.campaigns.monitoring.__main__ import execute, worker
from gleipnir.campaigns.monitoring.contract import Campaign
from gleipnir.campaigns.monitoring.evaluation import attach_inputs, id_metrics
from gleipnir.data.monitoring import file_hash, read_rows, write_json

ARM_TIMEOUT = 3 * 3600
MERGED_SLOT = Path("/tmp/gleipnir-merged/teacher-disagreement-sweep")


def launch(ctx: Campaign, stage: str) -> None:
    """Reuse the frozen original control weights; train only filtering arms."""
    if stage != "train" or "cached_control" not in ctx.config:
        worker(ctx, stage)
        return
    from gleipnir.campaigns.monitoring.training import validate_completion

    ctx.check()
    original = json.loads(ctx.input("cached_job").read_text())
    for key in [
        "seed",
        "learning_rate",
        "soft_loss_weight",
        "direct_loss_weight",
        "model_revision",
        "rank",
        "lora_alpha",
        "max_length",
        "gradient_checkpointing",
        "effective_batch_size",
    ]:
        if original[key] != ctx.job()[key]:
            raise ValueError(f"cached control recipe differs: {key}")
    metadata = json.loads(ctx.input("cached_metadata").read_text())
    validate_completion(metadata, ctx)
    filenames = {
        "cached_metadata": "causal_adapter/training_metadata.json",
        "cached_master_config": "causal_adapter/adapter_config.json",
        "cached_master": "causal_adapter/adapter_model.safetensors",
        "cached_export_config": "model/adapter_config.json",
        "cached_export": "model/adapter_model.safetensors",
        "cached_rebase": "model/rebase_manifest.json",
    }
    for name, filename in filenames.items():
        target = ctx.adapter / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise ValueError(f"refuse to overwrite reused control: {target}")
        shutil.copyfile(ctx.input(name), target)
    reuse = {
        "performed_this_run": False,
        "policy": "reuse_existing_unfiltered_sdpa",
        "original_campaign": "student_injection_awareness/4b/regular",
        "qualification": (
            "Same data/targets/seed/initialization and SDPA recipe; historical "
            "source/runtime/host differ. Fresh matched BF16 serving evaluation."
        ),
        "original_inputs": ctx.config["cached_control"]["inputs"],
    }
    write_json(ctx.output / "control_reuse.json", reuse)
    write_json(
        ctx.adapter / "complete.json",
        {
            "status": "reused_trained_control",
            "steps": 272,
            "epochs": 1.0,
            "training_performed_this_run": False,
            "manifest_sha256": file_hash(ctx.data / "manifest.json"),
            "master_sha256": file_hash(ctx.input("cached_master")),
            "serving_sha256": file_hash(ctx.input("cached_export")),
            "control_reuse": reuse,
        },
    )
    print("sweep_control_weights_reused", flush=True)


def retire(ctx: Campaign) -> None:
    """Retire only the scorer produced by this exact campaign."""
    active = ctx.serving / "server.json"
    if not active.exists():
        return
    receipt = json.loads(active.read_text())
    if receipt.get("log") != str(ctx.logs / "server.log"):
        raise ValueError("refuse to retire a scorer owned by another campaign")
    subprocess.run(
        [
            ctx.config["serving_python"],
            "-m",
            "experiments.b200_vllm031.runtime",
            "-m",
            "experiments.b200_vllm031.stop",
            "--archive-name",
            ctx.config["campaign_id"] + "_sweep_complete",
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}"},
        check=True,
        timeout=120,
    )


def release_merge(ctx: Campaign) -> None:
    """Verify and retire completed disposable weights before reusing one slot."""
    slot = Path(ctx.config["merged_model"])
    if slot != MERGED_SLOT or slot.is_symlink():
        raise ValueError("unexpected shared merge slot")
    if (ctx.serving / "server.json").exists():
        raise ValueError("retire scorer before replacing merged weights")
    complete = json.loads((ctx.output / "evaluation/complete.json").read_text())
    if not complete["parity_passed"] or complete["rows"] != 3012:
        raise ValueError("preserve merge until evaluation completes")
    artifact = json.loads((ctx.output / "merged_artifact.json").read_text())
    master = ctx.adapter / "causal_adapter/adapter_model.safetensors"
    if file_hash(master) != complete["master_sha256"]:
        raise ValueError("persistent master changed; preserve merged weights")
    for name, expected in artifact["files_sha256"].items():
        if file_hash(slot / name) != expected:
            raise ValueError("merged weights changed before slot retirement")
    manifest = slot / "merge_manifest.json"
    if json.loads(manifest.read_text()) != artifact:
        raise ValueError("merge manifest changed before slot retirement")
    archived = ctx.output / "archived_merge_manifest.json"
    shutil.copyfile(manifest, archived)
    write_json(
        ctx.output / "merged_slot_retired.json",
        {
            "destination": str(slot),
            "retired_at_unix": time.time(),
            "manifest_sha256": file_hash(archived),
            "master_preserved": True,
            "weights_preserved": False,
            "policy": "user_requested_shared_ephemeral_merge_slot",
            "note": (
                "Reconstruct from the preserved base and FP32 adapter; original "
                "stage receipts describe superseded files."
            ),
        },
    )
    shutil.rmtree(slot)


def terminate_owned(process: subprocess.Popen, owned: dict[int, float]) -> None:
    """Stop verified descendants on failure; never touch unrelated GPU processes."""
    if process.poll() is None:
        for child in psutil.Process(process.pid).children(recursive=True):
            owned[child.pid] = child.create_time()
    alive = []
    for pid, created in owned.items():
        try:
            child = psutil.Process(pid)
            if child.create_time() == created:
                child.terminate()
                alive.append(child)
        except psutil.NoSuchProcess:
            pass
    _, remaining = psutil.wait_procs(alive, timeout=15)
    for child in remaining:
        child.kill()
    _, remaining = psutil.wait_procs(remaining, timeout=15)
    if any(p.status() != psutil.STATUS_ZOMBIE for p in remaining):
        raise RuntimeError("owned campaign processes did not exit")


def summarize(manifest: dict) -> dict:
    """Verify complete predictions and recompute every arm's ID diagnostics."""
    cases = {}
    for name, spec in manifest["cases"].items():
        ctx = Campaign.load(ROOT, ROOT / spec["config"])
        ctx.check()
        complete = json.loads((ctx.output / "evaluation/complete.json").read_text())
        prediction_path = ctx.output / "evaluation/id.jsonl"
        if (
            not complete["parity_passed"]
            or complete["rows"] != 3012
            or complete["config_sha256"] != file_hash(ctx.config_path)
            or complete["files_sha256"]["id"] != file_hash(prediction_path)
        ):
            raise ValueError(f"incomplete or changed evaluation: {name}")
        scored = attach_inputs(read_rows(prediction_path), read_rows(ctx.input("id")))
        metrics = id_metrics(scored)
        saved = json.loads((ctx.output / "summary.json").read_text())["evaluations"][
            "id"
        ]
        if metrics != saved:
            raise ValueError(f"metric recomputation differs: {name}")
        cases[name] = {
            "delta": spec["delta"],
            "removed": spec["removed"],
            "retained": spec["retained"],
            "steps": ctx.config["expected_steps"],
            "tokens": ctx.config["expected_training_tokens"],
            "id": metrics,
            "predictions_sha256": file_hash(prediction_path),
            "training_performed_this_run": "cached_control" not in ctx.config,
        }
    baseline = cases["control"]["id"]["metrics"]["macro"]["macro"]
    lines = [
        "| Arm | Removed | Retained | Macro pAUROC@20 | Change | Macro AUROC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, case in cases.items():
        macro = case["id"]["metrics"]["macro"]["macro"]
        case["delta_vs_control"] = {
            k: macro[k] - baseline[k] for k in ["pauroc_at_20", "auroc"]
        }
        lines.append(
            f"| {name} | {case['removed']} | {case['retained']} | "
            f"{macro['pauroc_at_20']:.6f} | "
            f"{case['delta_vs_control']['pauroc_at_20']:+.6f} | "
            f"{macro['auroc']:.6f} |"
        )
    report = {
        "status": "complete",
        "selection": "fixed final one-epoch checkpoints",
        "ood_evaluated": False,
        "promotion": False,
        "cases": cases,
        "manifest_sha256": file_hash(OUTPUT / "manifest.json"),
    }
    write_json(OUTPUT / "summary.json", report)
    (OUTPUT / "comparison.md").write_text("\n".join(lines) + "\n")
    return report


def run() -> None:
    """Preflight all arms before the first GPU stage, then fail closed."""
    manifest = verify_preparation()
    status_path = OUTPUT / "status.json"
    if status_path.exists():
        raise ValueError("queue already launched; inspect existing receipts")
    for spec in manifest["cases"].values():
        ctx = Campaign.load(ROOT, ROOT / spec["config"])
        inventory = ctx.inventory(runtime=True)
        if not inventory["passed"]:
            raise ValueError(
                "prerequisite inventory failed: " + "\n".join(inventory["errors"])
            )
        ctx.prepare()
    logs = ROOT / "logs/runpod/teacher_disagreement_filtering" / OUTPUT.name
    logs.mkdir(parents=True, exist_ok=True)
    state = {
        "status": "running",
        "driver_pid": os.getpid(),
        "arms": {},
        "arm_timeout_seconds": ARM_TIMEOUT,
        "started_at_unix": time.time(),
    }
    write_json(status_path, state)
    for name, spec in manifest["cases"].items():
        verify_preparation()
        ctx = Campaign.load(ROOT, ROOT / spec["config"])
        command = [
            sys.executable,
            "-m",
            "experiments.teacher_disagreement_filtering.run",
            "--arm",
            name,
        ]
        with (logs / f"{name}.log").open("x") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=dict(os.environ),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        owned = {process.pid: psutil.Process(process.pid).create_time()}
        state["active_arm"] = name
        state["arms"][name] = {
            "status": "running",
            "pid": process.pid,
            "started_at_unix": time.time(),
        }
        write_json(status_path, state)
        print(f"sweep_arm_start {name} pid={process.pid}", flush=True)
        started = time.monotonic()
        try:
            while process.poll() is None:
                for child in psutil.Process(process.pid).children(recursive=True):
                    try:
                        owned[child.pid] = child.create_time()
                    except psutil.NoSuchProcess:
                        pass
                if time.monotonic() - started > ARM_TIMEOUT:
                    raise TimeoutError(f"{name} exceeded three-hour arm deadline")
                time.sleep(5)
            if process.returncode:
                raise RuntimeError(
                    f"{name} exited {process.returncode}; see {logs / (name + '.log')}"
                )
            retire(ctx)
            if name != list(manifest["cases"])[-1]:
                release_merge(ctx)
            state["arms"][name].update(status="complete", finished_at_unix=time.time())
            write_json(status_path, state)
        except BaseException as error:
            terminate_owned(process, owned)
            state.update(status="failed", error=str(error))
            state["arms"][name].update(status="failed", finished_at_unix=time.time())
            write_json(status_path, state)
            raise
        print(f"sweep_arm_complete {name}", flush=True)
    summarize(manifest)
    state.update(status="complete", active_arm=None, finished_at_unix=time.time())
    write_json(status_path, state)
    print("sweep_complete", flush=True)


def guarded_run() -> None:
    """Record preparation/launch failures as well as failures inside an arm."""
    try:
        run()
    except BaseException as error:
        failure = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "failed_at_unix": time.time(),
        }
        write_json(OUTPUT / "queue_failure.json", failure)
        status_path = OUTPUT / "status.json"
        if status_path.exists():
            state = json.loads(status_path.read_text())
            if state["status"] == "running":
                state.update(failure)
                write_json(status_path, state)
        else:
            write_json(status_path, failure)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.arm:
        manifest = verify_preparation()
        execute(
            Campaign.load(ROOT, ROOT / manifest["cases"][args.arm]["config"]),
            launch=launch,
        )
    elif args.summarize:
        summarize(verify_preparation())
    else:
        with (OUTPUT / "queue.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            guarded_run()
