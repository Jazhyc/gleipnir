"""Build the three-seed average, then merge, parity-check and score ID and OOD."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

from experiments.seed_average.average import build
from experiments.seed_average.prepare import AVERAGE, AVERAGE_OOD, OUT, ROOT
from experiments.teacher_disagreement_filtering.run import retire
from experiments.teacher_seed_variation.run import score
from gleipnir.adapters.rebase import rebase_adapter
from gleipnir.campaigns.monitoring.__main__ import execute, worker
from gleipnir.campaigns.monitoring.contract import Campaign
from gleipnir.data.monitoring import file_hash, write_json

STATUS = OUT / "status.json"


def build_average(ctx: Campaign) -> None:
    """The campaign's "train" stage: write the exact FP32 average, no training."""
    sources = ctx.config["seed_average_sources"]
    for seed, spec in sources.items():
        if file_hash(ROOT / spec["path"]) != spec["sha256"]:
            raise ValueError(f"seed {seed} FP32 master drift")
    adapter = ctx.adapter
    receipt = build(
        [Path(ROOT / s["path"]).parent for s in sources.values()],
        adapter / "causal_adapter",
    )
    write_json(
        adapter / "causal_adapter/training_metadata.json",
        {"seed_average": receipt, "sources": sources, "trained": False},
    )
    rebase_adapter(adapter / "causal_adapter", adapter / "model")
    write_json(
        adapter / "complete.json",
        {
            "status": "averaged",
            "steps": 0,
            "epochs": 1.0,
            "manifest_sha256": file_hash(ctx.data / "manifest.json"),
            "master_sha256": file_hash(
                adapter / "causal_adapter/adapter_model.safetensors"
            ),
            "serving_sha256": file_hash(adapter / "model/adapter_model.safetensors"),
            "sources": sources,
        },
    )
    print("seed_average_built", json.dumps(receipt), flush=True)


def launch(ctx: Campaign, stage: str) -> None:
    if stage == "train":
        build_average(ctx)
    else:
        worker(ctx, stage)


def run() -> None:
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).strip()
    if STATUS.exists():
        raise ValueError("seed-average queue already launched")
    ctx = Campaign.load(ROOT, ROOT / "data" / AVERAGE / "config.yaml")
    if gpu != ctx.config["gpu_uuid"]:
        raise ValueError("authorized GPU identity changed")
    state = {"status": "running", "driver_pid": os.getpid(), "started": time.time()}
    write_json(STATUS, state)
    try:
        execute(ctx, launch=launch)  # Average, master/merge parity, ID scoring.
        retire(ctx)
        state["id_complete"] = time.time()
        write_json(STATUS, state)
        score(AVERAGE_OOD)  # Re-merge (must be identical), then OOD scoring.
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(STATUS, state)
        raise
    state.update(status="complete", finished=time.time())
    write_json(STATUS, state)
    print("seed_average_complete", flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run()
