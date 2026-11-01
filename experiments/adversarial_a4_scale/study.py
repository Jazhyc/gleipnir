"""Serial dual-arm generation, with a shared completion barrier before scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import ROOT, config, digest, fixed, lock
from .dependency import arm_config, instructed_pairs
from .pipeline import generate, prepare, sealed, search_batch
from .selection import benchmark_pairs

ARMS = ("4b-v03", "27b-merged")


def validate_dual_seal(out: Path) -> dict:
    seal = json.loads((out / "generation_complete.json").read_text())
    for arm in ARMS:
        manifest = json.loads((out / arm / "manifest.json").read_text())
        records = sealed(out / arm, manifest)
        if seal[arm] != {"manifest": digest(manifest), "records": digest(records)}:
            raise ValueError("dual-arm completion seal changed")
    return seal


def validate_population_and_balance(out: Path, cfg: dict, pairs: dict) -> None:
    from experiments.adversarial_crossed_text.campaign import Campaign

    from .cost import check_gate, verify_population
    from .pipeline import endpoint, generation_records, messages, slots

    gate = check_gate(cfg)
    remaining = 0.0
    ledger = Campaign(cfg, out).latest()
    for arm in ARMS:
        dest = out / arm
        manifest = json.loads((dest / "manifest.json").read_text())
        arm_cfg = arm_config(cfg, arm)
        arm_pairs = instructed_pairs(pairs, arm_cfg)
        if manifest["config"] != arm_cfg:
            raise ValueError("prepared arm config changed")
        verify_population(gate, arm, manifest["selection"], arm_pairs, arm_cfg)
        records = generation_records(dest)
        for cid, task, _attempt in slots(manifest):
            if cid not in records:
                request_id = Campaign(cfg, out).request_id(dest, cid)
                if request_id in ledger:
                    raise RuntimeError(
                        "orphan campaign reservation requires reconciliation"
                    )
                remaining += endpoint(cfg["writer"]).worst_case_usd(
                    sum(
                        len(m["content"].encode())
                        for m in messages(arm_pairs[task]["backdoored"])
                    )
                )
    scope = sum(r["charge"] for r in ledger.values() if r["scope"] == "a4-writer")
    total = sum(r["charge"] for r in ledger.values() if r["api"] == "tinker")
    if (
        scope + remaining > min(140, cfg["budgets_usd"]["tinker"])
        or total + remaining > cfg["campaign"]["caps_usd"]["tinker"]
    ):
        raise RuntimeError("remaining aggregate run no longer fits campaign balance")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["prepare", "generate", "export-search"])
    p.add_argument(
        "--dependencies",
        type=Path,
        default=ROOT / "results/adversarial_a4_scale_dependencies_20261023",
    )
    args = p.parse_args()
    cfg = config()
    out = ROOT / cfg["output"]
    pairs = benchmark_pairs(cfg)
    with lock(out):
        if args.stage == "prepare":
            from experiments.adversarial_crossed_text.campaign import Campaign

            from .cost import check_gate

            gate = check_gate(cfg)
            latest = Campaign(cfg, out).latest().values()
            scope = sum(r["charge"] for r in latest if r["scope"] == "a4-writer")
            total = sum(r["charge"] for r in latest if r["api"] == "tinker")
            if (
                scope + gate["worst_case_generation_usd"]
                > min(140, cfg["budgets_usd"]["tinker"])
                or total + gate["worst_case_generation_usd"]
                > cfg["campaign"]["caps_usd"]["tinker"]
            ):
                raise RuntimeError("campaign balance no longer covers frozen run")
        if args.stage == "export-search":
            validate_dual_seal(out)
        seal = {}
        for arm in ARMS:
            arm_cfg = arm_config(cfg, arm)
            arm_pairs = instructed_pairs(pairs, arm_cfg)
            dest = out / arm
            if args.stage == "prepare":
                dependency = json.loads((args.dependencies / f"{arm}.json").read_text())
                manifest = prepare(dest, dependency, arm_pairs, arm_cfg)
                print(
                    json.dumps(
                        {
                            "arm": arm,
                            "tasks": len(manifest["selection"]["task_ids"]),
                            "estimate": manifest["generation_estimate"],
                        }
                    ),
                    flush=True,
                )
                continue
            manifest = json.loads((dest / "manifest.json").read_text())
            if args.stage == "generate":
                validate_population_and_balance(out, cfg, pairs)
                result = generate(dest, manifest, arm_pairs, _coordinated=True)
                print(json.dumps({"arm": arm, **result}), flush=True)
                if result["status"] != "complete":
                    return
                seal[arm] = {
                    "manifest": digest(manifest),
                    "records": digest(sealed(dest, manifest)),
                }
            else:
                batch = search_batch(dest, manifest, arm_pairs)
                print(json.dumps({"arm": arm, "jobs": len(batch["jobs"])}), flush=True)
        if args.stage == "generate":
            fixed(out / "generation_complete.json", seal)


if __name__ == "__main__":
    main()
