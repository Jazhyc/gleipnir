"""Evaluate the final FP4-trained adapter using bounded parity and BF16 vLLM."""

import argparse
import subprocess
import sys

from experiments.b200_fp4_full_training.campaign import (
    DATA,
    OUTPUT,
    ROOT,
    configuration,
    prepare,
)
from gleipnir.monitoring_campaign_evaluation import (
    EvaluationContext,
    reference,
    serving,
)
from gleipnir.monitoring_campaign_runtime import training_environment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=("reference", "vllm", "summary"), required=True
    )
    args = parser.parse_args()
    config = configuration()
    if args.backend == "summary":
        # Training provenance checks require its isolated kernel distributions;
        # serving intentionally excludes those versions from its Python path.
        subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.b200_fp4_full_training.comparison",
            ],
            cwd=ROOT,
            env=training_environment(ROOT, config["campaign_id"], native_fp4_mlp=True),
            check=True,
        )
        return
    manifest = prepare()
    context = EvaluationContext(
        DATA, OUTPUT, ("fp4",), manifest["template_sha256"], splits=("id",)
    )
    if args.backend == "reference":
        reference(context, config, "4b")
    elif args.backend == "vllm":
        serving(context, config, "4b")


if __name__ == "__main__":
    main()
