"""Rerun the 1K–1M C1 sweep with last-query attention in the final layer."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from experiments.b200_context_million.run import DEFAULT, Variant, run

EXPERIMENT = Path(__file__).parent
VARIANT = Variant(
    experiment=EXPERIMENT,
    result_group="b200_last_layer_trim",
    worker="experiments.b200_last_layer_trim.worker.LastLayerTrimWorker",
    sources=(
        *DEFAULT.sources,
        "experiments/b200_last_layer_trim/worker.py",
        "src/gleipnir/serving/last_query_attention.py",
    ),
    checks=(
        (
            "experiments.b200_last_layer_trim.native",
            "last_query.json",
            "GLEIPNIR_LAST_QUERY_VALIDATION",
        ),
    ),
    audits=("last_query_attention.json",),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--phase", action="append", choices=["native", "yarn"])
    args = parser.parse_args()
    if args.name in {"", ".", ".."} or Path(args.name).name != args.name:
        raise ValueError("run name must be a stem")
    asyncio.run(run(args.name, args.phase, VARIANT))


if __name__ == "__main__":
    main()
