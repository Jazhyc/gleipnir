"""Train one factorial teacher cell, score ID through the runner, then OOD."""

from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import time

from experiments.teacher_disagreement_filtering.run import retire
from experiments.teacher_precision_factorial.prepare import (
    CELLS,
    ROOT,
    campaign,
    scoring_name,
)
from experiments.teacher_seed_variation.run import score
from gleipnir.campaigns.monitoring.__main__ import execute
from gleipnir.campaigns.monitoring.contract import Campaign
from gleipnir.data.monitoring import write_json


def run(cell: str) -> None:
    status = ROOT / f"results/teacher_precision_factorial/status_{cell}.json"
    if status.exists():
        raise ValueError("cell already launched")
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).strip()
    ctx = Campaign.load(ROOT, ROOT / "data" / campaign(cell) / "config.yaml")
    if gpu != ctx.config["gpu_uuid"]:
        raise ValueError("authorized GPU identity changed")
    state = {"status": "running", "driver_pid": os.getpid(), "stages": {}}
    write_json(status, state)
    try:
        for name in (campaign(cell), scoring_name(cell)):
            state["stages"][name] = {"status": "running", "started": time.time()}
            write_json(status, state)
            if name == campaign(cell):
                inventory = ctx.inventory(runtime=True)
                if not inventory["passed"]:
                    raise ValueError("\n".join(inventory["errors"]))
                execute(ctx)
                retire(ctx)
            else:
                score(name)
            state["stages"][name].update(status="complete", finished=time.time())
            write_json(status, state)
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(status, state)
        raise
    state["status"] = "complete"
    write_json(status, state)
    print("cell_complete", cell, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", choices=sorted(CELLS), required=True)
    args = parser.parse_args()
    lock = ROOT / "results/teacher_precision_factorial/queue.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.cell)
