"""Run bounded original-FLA parity, then one persistent vLLM evaluation engine."""

from __future__ import annotations

import os
import subprocess
import sys

from experiments.monitoring_injection_disentanglement.prepare import ROOT
from gleipnir.monitoring_campaign_runtime import training_environment


def main() -> None:
    command = [
        sys.executable,
        "-m",
        "experiments.monitoring_injection_disentanglement.evaluate",
        "--stage",
    ]
    subprocess.run(
        [*command, "reference"],
        cwd=ROOT,
        env=training_environment(
            ROOT, "monitoring_injection_disentanglement_reference"
        ),
        check=True,
    )
    env = dict(os.environ)
    env.update(PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    subprocess.run([*command, "vllm"], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
