"""Evaluate only ID using the completed augmented adapter and BF16 serving."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from experiments.b200_augmented_training import campaign as original
from gleipnir.campaigns.monitoring.contract import Campaign
from gleipnir.campaigns.monitoring.evaluation import attach_inputs, optimized
from gleipnir.data.monitoring import file_hash, read_rows, write_json

CONFIG = Path(__file__).with_name("config.yaml")


class Completed(Campaign):
    @property
    def adapter(self) -> Path:
        return original.ADAPTER


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "score"), required=True)
    args = parser.parse_args()
    ctx = Campaign.load(original.ROOT, CONFIG)
    if args.stage == "prepare":
        ctx.prepare()
        for name in (
            "merged_artifact.json",
            "merged_parity.json",
            "canary_workload.jsonl",
        ):
            source = original.OUTPUT / name
            target = ctx.output / name
            if target.exists() and file_hash(target) != file_hash(source):
                raise ValueError("BF16 reference seed changed")
            target.write_bytes(source.read_bytes())
        print("bf16_id_prepared", flush=True)
        return
    ctx = Completed(ctx.root, ctx.config_path, ctx.config, ctx.source_root)
    ctx.check()
    write_json(ctx.output / "status.json", {"stage": "bf16_evaluation"})
    try:
        asyncio.run(optimized(ctx))
        current = read_rows(ctx.output / "evaluation/id.jsonl")
        control = read_rows(ctx.input("optimized_augmented_id"))
        attach_inputs(control, read_rows(ctx.input("id")))
        if [r["id"] for r in current] != [r["id"] for r in control]:
            raise ValueError("paired BF16/optimized population drift")
        import numpy as np

        x, y = (
            np.array([r["score"] for r in control]),
            np.array([r["score"] for r in current]),
        )
        paired = {
            "mean_absolute_difference": float(np.abs(y - x).mean()),
            "mean_score_delta": float((y - x).mean()),
            "correlation": float(np.corrcoef(x, y)[0, 1]),
            "half_threshold_flips": int(np.sum((x >= 0.5) != (y >= 0.5))),
        }
        write_json(ctx.output / "paired.json", paired)
        write_json(
            ctx.output / "status.json", {"stage": "complete", "rows": len(current)}
        )
        print("bf16_id_complete", paired, flush=True)
    except BaseException as error:
        write_json(ctx.output / "status.json", {"stage": "failed", "error": str(error)})
        raise


if __name__ == "__main__":
    main()
