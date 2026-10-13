"""Launch on the existing idle GPU; no capacity or volume lifecycle actions."""

import json
import subprocess
from pathlib import Path

import yaml

from gleipnir.campaigns.runtime import training_environment
from gleipnir.data.monitoring import write_json


def main():
    root = Path.cwd()
    config = yaml.safe_load(
        (root / "experiments/filtered_fp4_ood/config.yaml").read_text()
    )
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if active:
        raise RuntimeError("existing GPU has active compute processes")
    previous = json.loads((root / "results/qad-budget-sweep01/status.json").read_text())
    if previous["phase"] != "complete":
        raise RuntimeError("matched control is not complete")
    output = root / "results" / config["experiment"]
    output.mkdir(parents=True, exist_ok=True)
    log = root / "logs/runpod" / config["experiment"] / "all.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = training_environment(root, config["experiment"], native_fp4_mlp=True)
    with log.open("x") as stream:
        process = subprocess.Popen(
            [config["training_python"], "-m", "experiments.filtered_fp4_ood.run"],
            cwd=root,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    receipt = {"pid": process.pid, "log": str(log), "experiment": config["experiment"]}
    write_json(output / "all_launch.json", receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
