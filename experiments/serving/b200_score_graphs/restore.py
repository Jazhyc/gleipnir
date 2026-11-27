"""Restore the selected warm scorer without rerunning timing controls."""

import argparse
import asyncio
from pathlib import Path

from experiments.b200_monitor_score.run import EXPERIMENT, measure

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    if Path(args.name).name != args.name:
        raise ValueError("run name must be a directory stem")
    asyncio.run(
        measure(
            args.name,
            experiment=EXPERIMENT,
            result_group="b200_score_graphs",
            startup_only=True,
        )
    )
