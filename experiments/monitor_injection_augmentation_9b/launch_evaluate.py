"""Run the bounded master reference, then all scoring in one vLLM engine."""

import os
import subprocess
import sys

import yaml

from experiments.monitor_injection_augmentation_9b.prepare import CONFIG, ROOT
from gleipnir.monitoring_campaign_runtime import training_environment


def main() -> None:
    command = [
        sys.executable,
        "-m",
        "experiments.monitor_injection_augmentation_9b.evaluate",
        "--stage",
    ]
    campaign = yaml.safe_load(CONFIG.read_text())["campaign_id"]
    subprocess.run(
        [*command, "reference"],
        cwd=ROOT,
        env=training_environment(ROOT, campaign + "_reference"),
        check=True,
    )
    env = dict(os.environ)
    env.update(PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    subprocess.run([*command, "vllm"], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
