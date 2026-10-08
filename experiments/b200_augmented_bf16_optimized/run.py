"""Evaluate the completed BF16-trained adapter with selected optimized serving."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

from gleipnir.campaigns.monitoring.contract import Campaign
from gleipnir.campaigns.monitoring.evaluation import optimized
from gleipnir.data.monitoring import write_json

ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).with_name("config.yaml")


class Completed(Campaign):
    @property
    def adapter(self) -> Path:
        return self.root / self.config["completed_adapter"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "score"), required=True)
    args = parser.parse_args()
    ctx = Campaign.load(ROOT, CONFIG)
    if args.stage == "prepare":
        ctx.prepare()
        for target, source in (
            ("merged_artifact.json", "completed_merge"),
            ("merged_parity.json", "completed_merged_parity"),
            ("canary_workload.jsonl", "completed_canary"),
        ):
            path = ctx.output / target
            contents = ctx.input(source).read_bytes()
            if path.exists() and path.read_bytes() != contents:
                raise ValueError("prepared optimized reference changed")
            path.write_bytes(contents)
        print("optimized_followup_prepared", flush=True)
        return
    ctx = Completed(ctx.root, ctx.config_path, ctx.config, ctx.source_root)
    ctx.check()
    if (ctx.output / "server.json").exists():
        raise ValueError("optimized scoring already attempted; preserve its receipts")
    active = ctx.serving / "server.json"
    if active.exists():
        if json.loads(active.read_text()) != json.loads(
            ctx.input("resident_server").read_text()
        ):
            raise ValueError("recorded parent server identity changed")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.b200_vllm031.stop",
                "--archive-name",
                ctx.config["campaign_id"] + "_parent",
            ],
            cwd=ctx.root,
            env=dict(os.environ),
            check=True,
        )
    write_json(ctx.output / "status.json", {"stage": "optimized_evaluation"})
    try:
        asyncio.run(optimized(ctx))
        write_json(ctx.output / "status.json", {"stage": "complete", "rows": 12126})
    except BaseException as error:
        write_json(ctx.output / "status.json", {"stage": "failed", "error": str(error)})
        raise


if __name__ == "__main__":
    main()
