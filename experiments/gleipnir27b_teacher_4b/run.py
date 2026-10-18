"""Train and ID-score both Gleipnir-teacher arms in turn, releasing each merge."""

from __future__ import annotations

import fcntl
import os
import subprocess
import time

from experiments.distillation_temperature.recover import retire_scored_merge
from experiments.gleipnir27b_teacher_4b.prepare import ARMS, OUT, ROOT
from experiments.teacher_disagreement_filtering.run import retire
from gleipnir.campaigns.monitoring.__main__ import execute
from gleipnir.campaigns.monitoring.contract import Campaign
from gleipnir.data.monitoring import write_json

STATUS = OUT / "status.json"


def run() -> None:
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).strip()
    if STATUS.exists():
        raise ValueError("Gleipnir-teacher arms already launched")
    state = {"status": "running", "driver_pid": os.getpid(), "arms": {}}
    write_json(STATUS, state)
    for name, campaign in ARMS.items():
        arm = Campaign.load(ROOT, ROOT / "data" / campaign / "config.yaml")
        if gpu != arm.config["gpu_uuid"]:
            raise ValueError("authorized GPU identity changed")
        state["arms"][name] = {"status": "running", "started": time.time()}
        write_json(STATUS, state)
        try:
            execute(arm)  # Inventory, train, merge, parity and ID scoring.
            retire(arm)
            # Free the container disk; the FP32 master reconstructs the merge.
            retire_scored_merge(arm)
        except BaseException as error:
            state["arms"][name].update(
                status="failed", error=f"{type(error).__name__}: {error}"
            )
            state["status"] = "failed"
            write_json(STATUS, state)
            raise
        state["arms"][name].update(status="complete", finished=time.time())
        write_json(STATUS, state)
        print(f"gleipnir27b_teacher_arm_complete {name}", flush=True)
    state["status"] = "complete"
    write_json(STATUS, state)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run()
