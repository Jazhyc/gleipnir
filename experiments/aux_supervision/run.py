"""Train and ID-score the auxiliary-supervision arm, then release its merge."""

from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import time

from experiments.aux_supervision.prepare import ARM, ARMS, OUT, ROOT, receipt
from experiments.distillation_temperature.recover import retire_scored_merge
from experiments.teacher_disagreement_filtering.run import retire
from gleipnir.campaigns.monitoring.__main__ import execute
from gleipnir.campaigns.monitoring.contract import Campaign
from gleipnir.data.monitoring import write_json


def run(arm_id: str = ARM) -> None:
    status = receipt(arm_id, "status.json")
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).strip()
    if status.exists():
        raise ValueError("auxiliary-supervision arm already launched")
    arm = Campaign.load(ROOT, ROOT / "data" / arm_id / "config.yaml")
    if gpu != arm.config["gpu_uuid"]:
        raise ValueError("authorized GPU identity changed")
    state = {"status": "running", "driver_pid": os.getpid(), "started": time.time()}
    write_json(status, state)
    try:
        execute(arm)  # Inventory, train, merge, parity and ID scoring.
        retire(arm)
        # Free the container disk; the FP32 master reconstructs the merge.
        retire_scored_merge(arm)
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(status, state)
        raise
    state.update(status="complete", finished=time.time())
    write_json(status, state)
    print("aux_supervision_arm_complete", arm_id, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, default=ARM)
    args = parser.parse_args()
    receipt(args.arm, "status.json").parent.mkdir(parents=True, exist_ok=True)
    with (OUT / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.arm)
