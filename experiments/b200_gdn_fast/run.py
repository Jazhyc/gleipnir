"""C1 1K–1M sweep of the full adopted stack: last-query, gated and lean GDN."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
from pathlib import Path

from experiments.b200_context_million.run import run
from experiments.b200_gdn_route.run import variant

EXPERIMENT = Path(__file__).parent


def full_stack(cp_min_tokens: int = 8192):
    gated = variant("gated", cp_min_tokens)
    return dataclasses.replace(
        gated,
        experiment=EXPERIMENT,
        result_group="b200_gdn_fast",
        worker="experiments.b200_gdn_fast.worker.FastGatedTrimWorker",
        sources=(
            *gated.sources,
            "experiments/b200_gdn_fast/worker.py",
            "src/gleipnir/serving/gdn_fast.py",
        ),
        rpc_audits=(*gated.rpc_audits, ("gdn_fast_audit", "gdn_fast_audit.json")),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    if Path(args.name).name != args.name:
        raise ValueError("run name must be a stem")
    asyncio.run(run(args.name, None, full_stack()))


if __name__ == "__main__":
    main()
