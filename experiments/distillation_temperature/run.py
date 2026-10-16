"""Train and ID-score the frozen temperature arms serially on one GPU."""

from __future__ import annotations

import fcntl
import os
import subprocess
import time

from experiments.distillation_temperature.prepare import ARMS, ROOT
from experiments.teacher_disagreement_filtering.run import retire
from gleipnir.campaigns.monitoring.__main__ import execute
from gleipnir.campaigns.monitoring.contract import Campaign
from gleipnir.data.monitoring import write_json

STATUS = ROOT / "results/distillation_temperature/status.json"


def run() -> None:
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).strip()
    if STATUS.exists():
        raise ValueError("temperature queue already launched")
    state = {"status": "running", "driver_pid": os.getpid(), "stages": {}}
    write_json(STATUS, state)
    try:
        for name in ARMS:
            arm = Campaign.load(ROOT, ROOT / "data" / name / "config.yaml")
            if gpu != arm.config["gpu_uuid"]:
                raise ValueError("authorized GPU identity changed")
            state["stages"][name] = {"status": "running", "started": time.time()}
            write_json(STATUS, state)
            execute(arm)  # Inventory, train, merge, parity and ID scoring.
            retire(arm)
            state["stages"][name].update(status="complete", finished=time.time())
            write_json(STATUS, state)
            print("distillation_temperature_stage_complete", name, flush=True)
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(STATUS, state)
        raise
    state["status"] = "complete"
    write_json(STATUS, state)
    print("distillation_temperature_complete", flush=True)


if __name__ == "__main__":
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    with (STATUS.parent / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run()
