"""Run training-source serving parity and canonical ID in one persistent engine."""

from __future__ import annotations

import argparse

from experiments.monitoring_hard_labels.prepare import (
    DATA,
    FRACTIONS,
    OUTPUT,
    configuration,
    verify_preparation,
)
from gleipnir.monitoring_campaign_evaluation import (
    EvaluationContext,
    reference,
    serving,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("reference", "vllm"), required=True)
    args = parser.parse_args()
    config = configuration()
    manifest = verify_preparation()
    context = EvaluationContext(
        DATA, OUTPUT, tuple(FRACTIONS), manifest["template_sha256"], splits=("id",)
    )
    if args.backend == "reference":
        reference(context, config, "4b")
    else:
        serving(context, config, "4b")


if __name__ == "__main__":
    main()
