"""Bounded source and memory preflights; never silently start full training."""

import json
import subprocess
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from gleipnir.monitoring_systems_screen import atomic_write_json


def main():
    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).parent.resolve())
    ):
        config = OmegaConf.to_container(compose(config_name="preflight"), resolve=True)
    root = Path("results/monitoring_branching/hybrid_preflight_v1")
    execute(config, root)


def execute(config: dict, root: Path) -> None:
    """Run the frozen diagnostic recipe and stop before any training on failure."""
    if (
        config["prefix_weight"] != 0.1
        or config["alignment"] != 64
        or any(
            config[key] is not True
            for key in (
                "checkpoint_segments",
                "independent_endpoint",
                "fp32_head",
                "two_gpu",
            )
        )
        or config["training_authorized_after_this_only"] is not False
    ):
        raise ValueError("preflight recipe drift")
    root.mkdir(parents=True, exist_ok=False)
    atomic_write_json(root / "config.json", config)
    status = {"state": "running", "completed": [], "active": None}
    try:
        for kind in ("parity", "memory"):
            for parent in config[f"{kind}_parents"]:
                name = f"{kind}-{parent}"
                status["active"] = name
                atomic_write_json(root / "status.json", status)
                command = [
                    sys.executable,
                    "-u",
                    "-m",
                    "experiments.monitoring_branching.gpu_canary",
                    "--parent",
                    str(parent),
                    "--alignment",
                    "64",
                    "--checkpoint-segments",
                    "--independent-endpoint",
                    "--fp32-head",
                    "--two-gpu",
                    "--output",
                    str(root / f"{name}.json"),
                ]
                if kind == "memory":
                    command.append("--stress-only")
                with (root / f"{name}.log").open("x") as log:
                    subprocess.run(
                        command, stdout=log, stderr=subprocess.STDOUT, check=True
                    )
                result = json.loads((root / f"{name}.json").read_text())
                if not result["passed"]:
                    raise RuntimeError(f"preflight failed: {name}")
                status["completed"].append(name)
                print(name, json.dumps(result), flush=True)
        status.update(state="complete", active=None)
    except BaseException as error:
        status.update(state="failed", error=repr(error))
        raise
    finally:
        atomic_write_json(root / "status.json", status)


if __name__ == "__main__":
    main()
