"""Rerun the 1K–1M C1 sweep with the final decoder layer trimmed to pooled rows."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
from pathlib import Path

from experiments.b200_context_million.run import DEFAULT, Variant, run

EXPERIMENT = Path(__file__).parent
WORKERS = "experiments.b200_last_layer_trim.worker"
LAST = Variant(
    experiment=EXPERIMENT,
    result_group="b200_last_layer_trim",
    worker=f"{WORKERS}.LastLayerTrimWorker",
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
    rpc_audits=(("last_query_audit", "last_query_audit.json"),),
)
FINISHED = dataclasses.replace(LAST, worker=f"{WORKERS}.FinishedTrimWorker")
TAIL = dataclasses.replace(
    FINISHED,
    worker=f"{WORKERS}.TailTrimWorker",
    server="experiments.b200_last_layer_trim.server",
    sources=(
        *FINISHED.sources,
        "experiments/b200_last_layer_trim/server.py",
        "src/gleipnir/serving/last_query_tail.py",
    ),
)
VARIANTS = {"last": LAST, "finished": FINISHED, "tail": TAIL}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="last")
    parser.add_argument("--phase", action="append", choices=["native", "yarn"])
    args = parser.parse_args()
    if args.name in {"", ".", ".."} or Path(args.name).name != args.name:
        raise ValueError("run name must be a stem")
    asyncio.run(run(args.name, args.phase, VARIANTS[args.variant]))


if __name__ == "__main__":
    main()
