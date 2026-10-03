"""Original-FLA reference, then a persistent vLLM engine for all fresh cells."""

from __future__ import annotations

import os
import subprocess
import sys

from experiments.monitoring_injection_removal.prepare import ROOT
from gleipnir.monitoring_campaign_runtime import training_environment


def main() -> None:
    command = [
        sys.executable,
        "-m",
        "experiments.monitoring_injection_removal.evaluate",
        "--stage",
    ]
    subprocess.run(
        [*command, "reference"],
        cwd=ROOT,
        env=training_environment(ROOT, "monitoring_injection_removal_reference"),
        check=True,
    )
    env = dict(os.environ)
    env.update(PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    subprocess.run([*command, "vllm"], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
