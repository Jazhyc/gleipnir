"""Train the hard-label 27B campaign (which scores ID), then score OOD.

The training campaign runs through the shared monitoring campaign; the OOD
context reuses the seed-variation merge and BF16 scoring stages, as the soft
27B and the 4B hard seeds did.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import time

from experiments.teachers.teacher_disagreement_filtering.run import retire
from experiments.teachers.teacher_seed_variation.run import score
from experiments.training.qwen38_27b_hard_labels.prepare import (
    ARM,
    OOD_CONTEXT,
    ROOT,
)
from gleipnir.campaigns.monitoring.__main__ import execute
from gleipnir.campaigns.monitoring.contract import Campaign
from gleipnir.data.monitoring import write_json

STATUS = ROOT / "results/qwen38_27b_hard_labels/status.json"


def run(resume: bool = False) -> None:
    """``resume`` continues a failed queue, skipping stages already complete."""
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).strip()
    if STATUS.exists() and not resume:
        raise ValueError("hard-label 27B queue already launched")
    state = {"status": "running", "driver_pid": os.getpid(), "stages": {}}
    if resume:
        previous = json.loads(STATUS.read_text())
        if previous["status"] != "failed":
            raise ValueError("only a failed queue can be resumed")
        state["stages"] = {
            k: v for k, v in previous["stages"].items() if v["status"] == "complete"
        }
        state["resumed_after"] = previous.get("error")
    write_json(STATUS, state)
    try:
        arm = Campaign.load(ROOT, ROOT / "data" / ARM / "config.yaml")
        if gpu != arm.config["gpu_uuid"]:
            raise ValueError("authorized GPU identity changed")
        for stage in (ARM, OOD_CONTEXT):
            if state["stages"].get(stage, {}).get("status") == "complete":
                continue
            state["stages"][stage] = {"status": "running", "started": time.time()}
            write_json(STATUS, state)
            if stage == ARM:
                inventory = arm.inventory(runtime=True)
                if not inventory["passed"]:
                    raise ValueError("\n".join(inventory["errors"]))
                execute(arm, resume=resume)
                retire(arm)
            else:
                score(stage)
            state["stages"][stage].update(status="complete", finished=time.time())
            write_json(STATUS, state)
            print("qwen38_27b_hard_labels_stage_complete", stage, flush=True)
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(STATUS, state)
        raise
    state["status"] = "complete"
    write_json(STATUS, state)
    print("qwen38_27b_hard_labels_complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    with (STATUS.parent / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.resume)
