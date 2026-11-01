"""CLI for offline preparation and parent-owned scoring transport."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import ROOT, config, fixed, lock, read
from .dependency import arm_config, instructed_pairs
from .pipeline import (
    generate,
    prepare,
    rank,
    report,
    search_batch,
    transfer_batch,
    validate_scores,
)
from .selection import benchmark_pairs, recount


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "recount",
            "prepare",
            "generate",
            "export-search",
            "rank",
            "export-transfer",
            "import-transfer",
            "report",
        ],
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--monitor", choices=["4b-v03", "27b-merged"], default="4b-v03")
    parser.add_argument("--dependency", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--monitor-identity", type=Path)
    parser.add_argument("--instruction", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    cfg = arm_config(config(), args.monitor)
    out = args.out or ROOT / cfg["output"] / args.monitor
    if args.command == "recount":
        result = recount(cfg)
        # Print counts/hashes only; task IDs remain in ignored output if requested.
        if args.out:
            with lock(out):
                fixed(out / "historical_recount.json", result)
        print(
            json.dumps(
                {k: v for k, v in result.items() if not k.endswith("task_ids")},
                indent=2,
            )
        )
        return
    if args.command == "prepare":
        if args.dependency is None:
            parser.error(
                "--dependency is required; fresh selection has no historical fallback"
            )
        result = prepare(
            out,
            json.loads(args.dependency.read_text()),
            instructed_pairs(benchmark_pairs(cfg), cfg),
            cfg,
        )
        print(
            json.dumps(
                {
                    "selected_tasks": len(result["selection"]["task_ids"]),
                    "generation_estimate": result["generation_estimate"],
                    "out": str(out),
                }
            )
        )
        return
    manifest = json.loads((out / "manifest.json").read_text())
    pairs = instructed_pairs(benchmark_pairs(cfg), cfg)
    if args.command == "generate":
        if cfg.get("dual_arm_barrier"):
            parser.error(
                "full generation requires "
                "python -m experiments.adversarial_a4_scale.study generate"
            )
        responder = (
            (
                lambda _: json.dumps(
                    [{"before_line": 1, "comment": "Implementation notes."}]
                )
            )
            if args.mock
            else None
        )
        result = generate(out, manifest, pairs, responder=responder)
    elif args.command == "export-search":
        batch = search_batch(out, manifest, pairs)
        result = {"jobs": len(batch["jobs"]), "artifact": str(out / "search_jobs.json")}
    elif args.command == "rank":
        if args.receipt is None:
            parser.error("--receipt required")
        ranked = rank(out, manifest, pairs, json.loads(args.receipt.read_text()))
        result = {"tasks_ranked": len(ranked["winners"])}
    elif args.command == "export-transfer":
        if args.monitor_identity is None:
            parser.error("--monitor-identity required")
        batch = transfer_batch(
            out,
            manifest,
            pairs,
            json.loads(args.monitor_identity.read_text()),
            args.instruction.read_text() if args.instruction else None,
        )
        result = {
            "jobs": len(batch["jobs"]),
            "monitor": batch["monitor_identity"]["monitor"],
        }
    elif args.command == "import-transfer":
        if args.receipt is None:
            parser.error("--receipt required")
        receipt = json.loads(args.receipt.read_text())
        monitor = receipt["monitor_identity"]["monitor"]
        allowed = [manifest["selection"]["monitor_identity"]["monitor"]] + cfg[
            "monitor_contracts"
        ]["transfer"]
        if monitor not in allowed:
            parser.error("unregistered monitor")
        batch = json.loads((out / f"transfer_{monitor}_jobs.json").read_text())
        validate_scores(batch, receipt)
        with lock(out):
            fixed(out / f"transfer_{monitor}_scores.json", receipt)
        result = {"imported": monitor}
    else:
        result = report(
            out,
            manifest,
            audits=read(args.audit) if args.audit else None,
            final=args.final,
        )
        # Reports can be regenerated as transfer and audit receipts arrive.
        print(json.dumps(result, indent=2))
        return
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
