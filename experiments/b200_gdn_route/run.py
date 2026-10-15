"""C1 sweep of FlashInfer GDN prefill routes on the adopted last-query recipe."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
from pathlib import Path

from experiments.b200_context_million.run import run
from experiments.b200_last_layer_trim.run import LAST

EXPERIMENT = Path(__file__).parent
BASE = dataclasses.replace(LAST, experiment=EXPERIMENT, result_group="b200_gdn_route")


def variant(route: str, cp_min_tokens: int | None) -> object:
    if route == "off":
        if cp_min_tokens is not None:
            raise ValueError("the non-CP route takes no threshold")
        return dataclasses.replace(
            BASE, environment={"GLEIPNIR_FLASHINFER_GDN_CP": "off"}
        )
    if cp_min_tokens is None:
        raise ValueError("the gated route needs --cp-min-tokens")
    return dataclasses.replace(
        BASE,
        worker="experiments.b200_gdn_route.worker.GatedTrimWorker",
        sources=(
            *BASE.sources,
            "experiments/b200_gdn_route/worker.py",
            "src/gleipnir/serving/gdn_route.py",
        ),
        environment={"GLEIPNIR_GDN_CP_MIN_TOKENS": str(cp_min_tokens)},
        rpc_audits=(*BASE.rpc_audits, ("gdn_route_audit", "gdn_route_audit.json")),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--route", choices=["off", "gated"], required=True)
    parser.add_argument("--cp-min-tokens", type=int)
    args = parser.parse_args()
    if Path(args.name).name != args.name:
        raise ValueError("run name must be a stem")
    asyncio.run(run(args.name, None, variant(args.route, args.cp_min_tokens)))


if __name__ == "__main__":
    main()
