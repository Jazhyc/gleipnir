"""Train and ID-score the hard-label arm, then release its merged weights."""

from __future__ import annotations

import fcntl
import os
import subprocess
import time

from experiments.distillation_temperature.recover import retire_scored_merge
from experiments.filtered_hard_labels.prepare import ARM, ROOT
from experiments.teacher_disagreement_filtering.run import retire
from gleipnir.campaigns.monitoring.__main__ import execute
from gleipnir.campaigns.monitoring.contract import Campaign
from gleipnir.data.monitoring import write_json

STATUS = ROOT / "results/filtered_hard_labels/status.json"


def run() -> None:
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).strip()
    if STATUS.exists():
        raise ValueError("hard-label arm already launched")
    arm = Campaign.load(ROOT, ROOT / "data" / ARM / "config.yaml")
    if gpu != arm.config["gpu_uuid"]:
        raise ValueError("authorized GPU identity changed")
    state = {"status": "running", "driver_pid": os.getpid(), "started": time.time()}
    write_json(STATUS, state)
    try:
        execute(arm)  # Inventory, train, merge, parity and ID scoring.
        retire(arm)
        # Free the container disk; the FP32 master reconstructs the merge.
        retire_scored_merge(arm)
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(STATUS, state)
        raise
    state.update(status="complete", finished=time.time())
    write_json(STATUS, state)
    print("filtered_hard_labels_complete", flush=True)


if __name__ == "__main__":
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    with (STATUS.parent / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run()
