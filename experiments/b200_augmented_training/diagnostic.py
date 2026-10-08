"""User-authorized diagnostic of the completed adapter after failed serving parity."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from experiments.b200_augmented_training import campaign as original
from experiments.b200_augmented_training.evaluate import summarize as original_summary
from gleipnir.campaigns.monitoring.contract import Campaign
from gleipnir.campaigns.monitoring.evaluation import optimized
from gleipnir.data.monitoring import file_hash, read_rows, write_json


class Diagnostic(Campaign):
    """Read the completed adapter without modifying its training job or receipts."""

    @property
    def adapter(self) -> Path:
        return original.ADAPTER

    def check(self) -> dict:
        original.check_binding(original.configuration())
        return super().check()


def prepare(config_path: Path) -> Campaign:
    """Freeze a separate evaluation contract; reuse passed merge/reference artifacts."""
    ctx = Campaign.load(original.ROOT, config_path)
    original.check_binding(original.configuration())
    if not json.loads(original.OUTPUT.joinpath("merged_parity.json").read_text())[
        "passed"
    ]:
        raise ValueError("diagnostic still requires passed master/merged parity")
    ctx.prepare()
    for name in ("merged_parity.json", "merged_artifact.json", "canary_workload.jsonl"):
        target = ctx.output / name
        source = original.OUTPUT / name
        if target.exists():
            if file_hash(target) != file_hash(source):
                raise ValueError(f"diagnostic seed artifact drift: {name}")
        else:
            target.write_bytes(source.read_bytes())
    write_json(
        ctx.output / "authorization.json",
        {
            "instruction": (
                "Run optimized scoring as a failed-parity diagnostic (Recommended)"
            ),
            "scope": (
                "ID/APPS on the fixed completed adapter; "
                "no new optimizer updates or promotion"
            ),
            "original_failed_parity_sha256": file_hash(
                original.OUTPUT / "optimized_parity.json"
            ),
            "original_manifest_sha256": file_hash(original.DATA / "manifest.json"),
        },
    )
    return ctx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("prepare", "score", "summarize"), required=True
    )
    args = parser.parse_args()
    if args.stage == "prepare":
        ctx = prepare(args.config)
        print("diagnostic_prepared", ctx.output, flush=True)
        return
    plain = Campaign.load(original.ROOT, args.config)
    ctx = Diagnostic(plain.root, plain.config_path, plain.config, plain.source_root)
    ctx.check()
    if not ctx.config["evaluation"].get("failed_parity_diagnostic", False):
        raise ValueError("diagnostic requires an explicit frozen diagnostic flag")
    if args.stage == "score":
        if (ctx.output / "evaluation/complete.json").exists():
            raise ValueError(
                "diagnostic already completed; do not repeat held-out scoring"
            )
        write_json(ctx.output / "status.json", {"stage": "optimized_diagnostic"})
        try:
            asyncio.run(optimized(ctx))
        except BaseException as error:
            write_json(
                ctx.output / "status.json", {"stage": "failed", "error": str(error)}
            )
            raise
    populations = {
        s: read_rows(ctx.output / "evaluation" / f"{s}.jsonl") for s in ctx.splits
    }
    summary = original_summary(populations, original.configuration())
    summary.update(
        evaluation_scope="failed_parity_diagnostic",
        parity=json.loads((ctx.output / "optimized_parity.json").read_text()),
        diagnostic_manifest_sha256=file_hash(ctx.data / "manifest.json"),
    )
    write_json(ctx.output / "summary_original_contract.json", summary)
    write_json(
        ctx.output / "status.json",
        {
            "stage": "complete",
            "rows": 12126,
            "evaluation_scope": "failed_parity_diagnostic",
        },
    )
    print("diagnostic_complete", flush=True)


if __name__ == "__main__":
    main()
