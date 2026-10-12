"""Launch diagnostics or the automatic campaign on existing authorized capacity."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import yaml

from gleipnir.campaigns.runtime import training_environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["backward", "all"], default="all")
    args = parser.parse_args()
    root = Path.cwd()
    config = yaml.safe_load(
        (root / "experiments/native_fp8_qat/config.yaml").read_text()
    )
    experiment = config["experiment"]
    output = root / "results" / experiment
    output.mkdir(parents=True, exist_ok=True)
    if args.stage == "all":
        old = root / "results/qat-training-optimization02"
        phase = json.loads((old / "status.json").read_text())["phase"]
        if phase not in {"idle", "worker_stopped"}:
            raise RuntimeError("previous resident worker is busy")
        (old / "stop_worker").touch()
        before = time.monotonic()
        while (
            json.loads((old / "status.json").read_text())["phase"] != "worker_stopped"
        ):
            if time.monotonic() - before > 30:
                raise TimeoutError("resident worker retirement")
            time.sleep(1)
        # GPU context release can follow the stop receipt by a brief interval.
        pid = int((old / "worker.pid").read_text())
        while Path(f"/proc/{pid}/stat").exists():
            if Path(f"/proc/{pid}/stat").read_text().split()[2] == "Z":
                break
            if time.monotonic() - before > 45:
                raise TimeoutError("resident CUDA context release")
            time.sleep(1)
    module = "backward_probe" if args.stage == "backward" else "run"
    log = root / "logs/runpod" / experiment / f"{args.stage}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = training_environment(root, experiment, native_fp4_mlp=True)
    with log.open("x") as stream:
        process = subprocess.Popen(
            [
                "/tmp/gleipnir-serving-runtime/venv/bin/python",
                "-m",
                f"experiments.native_fp8_qat.{module}",
            ],
            cwd=root,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    write = {"pid": process.pid, "stage": args.stage, "log": str(log)}
    (output / f"{args.stage}_launch.json").write_text(
        json.dumps(write, indent=2) + "\n"
    )
    print(json.dumps(write))


if __name__ == "__main__":
    main()
