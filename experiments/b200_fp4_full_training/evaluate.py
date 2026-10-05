"""Evaluate the final FP4-trained adapter using bounded parity and BF16 vLLM."""

import argparse

from experiments.b200_fp4_full_training.campaign import (
    DATA,
    OUTPUT,
    configuration,
    prepare,
    write_comparison,
)
from gleipnir.monitoring_campaign_evaluation import (
    EvaluationContext,
    reference,
    serving,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=("reference", "vllm", "summary"), required=True
    )
    args = parser.parse_args()
    config = configuration()
    manifest = prepare()
    context = EvaluationContext(
        DATA, OUTPUT, ("fp4",), manifest["template_sha256"], splits=("id",)
    )
    if args.backend == "reference":
        reference(context, config, "4b")
    elif args.backend == "vllm":
        serving(context, config, "4b")
    else:
        result = write_comparison()
        print(result["macro_differences"], flush=True)


if __name__ == "__main__":
    main()
