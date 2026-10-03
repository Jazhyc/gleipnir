"""Serialize bounded original-FLA parity reference and persistent vLLM scoring."""

import os
import subprocess
import sys

from experiments.judge_injection_joint.prepare import ROOT, configuration
from gleipnir.monitoring_campaign_runtime import training_environment


def main() -> None:
    command = [
        sys.executable,
        "-m",
        "experiments.judge_injection_joint.evaluate",
        "--stage",
    ]
    subprocess.run(
        [*command, "reference"],
        cwd=ROOT,
        env=training_environment(ROOT, configuration()["campaign_id"] + "_reference"),
        check=True,
    )
    env = dict(os.environ)
    env.update(PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    subprocess.run([*command, "vllm"], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
