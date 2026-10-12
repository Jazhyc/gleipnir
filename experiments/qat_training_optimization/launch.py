"""Start one detached resident worker with shared pinned training caches."""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import yaml

from gleipnir.campaigns.runtime import training_environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    config = yaml.safe_load(
        (root / "experiments/qat_training_optimization/config.yaml").read_text()
    )
    experiment = config["experiment"]
    output = root / "results" / experiment
    output.mkdir(parents=True, exist_ok=True)
    pid_file = output / "worker.pid"
    if pid_file.exists():
        status_path = output / "status.json"
        status = json.loads(status_path.read_text())
        pid = int(pid_file.read_text())
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            alive = False
        else:
            alive = Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
        if not args.retry_failed or alive or status["phase"] != "failed":
            raise RuntimeError("existing worker receipt; inspect before launching")
        archive = output / f"failed_attempt_{pid}"
        archive.mkdir(exist_ok=False)
        shutil.move(pid_file, archive / pid_file.name)
        shutil.copy2(status_path, archive / status_path.name)
        old_log = root / "logs/runpod" / experiment / "worker.log"
        shutil.move(old_log, archive / "worker.log")
    log = root / "logs/runpod" / experiment / "worker.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = training_environment(root, experiment, native_fp4_mlp=True)
    shutil.copy2(Path(__file__).with_name("worker.py"), output / "worker_source.py")
    with log.open("x") as stream:
        process = subprocess.Popen(
            [
                "/tmp/gleipnir-serving-runtime/venv/bin/python",
                "-m",
                "experiments.qat_training_optimization.worker",
            ],
            cwd=root,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_file.write_text(str(process.pid))
    print(json.dumps({"pid": process.pid, "log": str(log)}))


if __name__ == "__main__":
    main()
