"""Train the three hard-label seeds in order, each followed by its OOD scoring.

Training campaigns run (and score ID) through the shared monitoring campaign;
OOD contexts reuse the seed-variation merge and BF16 scoring stages, which
load any scoring context by name.
"""

from __future__ import annotations

import fcntl
import os
import subprocess
import time

from experiments.hard_label_seeds.prepare import ROOT, SEEDS, campaign, ood_context
from experiments.teacher_disagreement_filtering.run import retire
from experiments.teacher_seed_variation.run import score
from gleipnir.campaigns.monitoring.__main__ import execute
from gleipnir.campaigns.monitoring.contract import Campaign
from gleipnir.data.monitoring import write_json

STATUS = ROOT / "results/hard_label_seeds/status.json"


def run() -> None:
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).strip()
    if STATUS.exists():
        raise ValueError("hard-label seed queue already launched")
    state = {"status": "running", "driver_pid": os.getpid(), "stages": {}}
    write_json(STATUS, state)
    try:
        for seed_number in SEEDS:
            name = campaign(seed_number)
            seed = Campaign.load(ROOT, ROOT / "data" / name / "config.yaml")
            if gpu != seed.config["gpu_uuid"]:
                raise ValueError("authorized GPU identity changed")
            for stage in (name, ood_context(seed_number)):
                state["stages"][stage] = {"status": "running", "started": time.time()}
                write_json(STATUS, state)
                if stage == name:
                    inventory = seed.inventory(runtime=True)
                    if not inventory["passed"]:
                        raise ValueError("\n".join(inventory["errors"]))
                    execute(seed)
                    retire(seed)
                else:
                    score(stage)
                state["stages"][stage].update(status="complete", finished=time.time())
                write_json(STATUS, state)
                print("hard_label_seeds_stage_complete", stage, flush=True)
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(STATUS, state)
        raise
    state["status"] = "complete"
    write_json(STATUS, state)
    print("hard_label_seeds_complete", flush=True)


if __name__ == "__main__":
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    with (STATUS.parent / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run()
