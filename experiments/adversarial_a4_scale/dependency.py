"""Read-only bridge from sealed Step 1 artifacts to per-monitor A4 cohorts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gleipnir.adversarial.contract import join_prompt, split_prompt
from gleipnir.adversarial.eval_sets import file_sha256

from .common import ROOT, config, digest, fixed, read, sha, unique
from .selection import benchmark_pairs, select

BACKEND = "transformers_peft_bf16_sdpa_fla052"


def arm_config(cfg: dict, monitor: str) -> dict:
    cfg = json.loads(json.dumps(cfg))
    cfg["monitor_contracts"]["search"] = cfg["monitor_contracts"]["search_arms"][
        monitor
    ]
    cfg["monitor_contracts"]["transfer"] = [
        m for m in cfg["monitor_contracts"]["all_monitors"] if m != monitor
    ]
    return cfg


def instruction(cfg: dict, name: str) -> str:
    spec = cfg["instructions"][name]
    text = (ROOT / spec["path"]).read_text()
    if sha(text) != spec["sha256"]:
        raise ValueError("instruction checksum changed")
    return text


def instructed_pairs(pairs: dict, cfg: dict) -> dict:
    head = instruction(cfg, cfg["monitor_contracts"]["search"]["instruction"])
    return {
        t: {
            c: {**r, "prompt": join_prompt(head, split_prompt(r["prompt"]).trajectory)}
            for c, r in codes.items()
        }
        for t, codes in pairs.items()
    }


def build(step1: Path, monitor: str, cfg: dict) -> dict:
    cfg = arm_config(cfg, monitor)
    arm = cfg["monitor_contracts"]["search"]
    name = f"{monitor}_{arm['instruction']}"
    sealed = json.loads((step1 / "variants.manifest.json").read_text())
    if sealed["dry_run"]:
        raise ValueError("mock Step 1 is not eligible")
    for filename, expected in sealed["files"].items():
        if file_sha256(step1 / filename) != expected:
            raise ValueError("Step 1 artifact seal mismatch")
    for filename, expected in sealed["sources"].items():
        if file_sha256(ROOT / filename) != expected:
            raise ValueError("Step 1 source seal mismatch")
    selection = json.loads((step1 / "selection.json").read_text())
    runtime_path = step1 / "scores" / f"{name}.heldout.runtime.json"
    canary_path = step1 / "scores" / f"{name}.heldout.canary.json"
    runtime = json.loads(runtime_path.read_text())
    canary = json.loads(canary_path.read_text())
    if (
        runtime["dry_run"]
        or not canary["passed"]
        or runtime["arm"] != {"monitor": monitor, "prompt": arm["instruction"]}
    ):
        raise ValueError("Step 1 runtime/canary contract mismatch")
    if runtime["variants_sha256"] != sealed["files"]["variants.jsonl"] or runtime[
        "variants_manifest_sha256"
    ] != file_sha256(step1 / "variants.manifest.json"):
        raise ValueError("Step 1 score runtime belongs to another build")
    scorer = runtime["scorer"]
    if (
        scorer["dtype"] != "bfloat16"
        or not scorer["fla_available"]
        or scorer["packages"]["flash-linear-attention"] != "0.5.2"
    ):
        raise ValueError("Step 1 backend mismatch")
    variants = unique(read(step1 / "variants.jsonl"))
    # 4B includes two passes of calibration. Accept exact repeats only.
    scores = {}
    score_path = step1 / "scores" / f"{name}.jsonl"
    for r in read(score_path):
        if r["id"] in scores and r != scores[r["id"]]:
            raise ValueError("conflicting Step 1 score rows")
        scores[r["id"]] = r
    eligible = [
        v
        for v in variants.values()
        if v["metadata"]["text"] == "none"
        and v["metadata"]["split"] == "heldout"
        and v["metadata"]["set"] == "main"
    ]
    tasks = sorted({str(v["metadata"]["task_id"]) for v in eligible})
    if len(eligible) != 2 * len(tasks) or not set(tasks) <= set(selection["heldout"]):
        raise ValueError("incomplete Step 1 heldout pairs")
    calibration = [
        v for v in variants.values() if v["metadata"]["set"] == "calibration"
    ]
    if len(calibration) != 221 or {
        str(v["metadata"]["task_id"]) for v in calibration
    } != set(selection["calibration"]):
        raise ValueError("need all 221 Step 1 calibration rows")
    pairs = instructed_pairs(benchmark_pairs(cfg), cfg)
    baseline = []
    for v in eligible + calibration:
        m, r = v["metadata"], scores[v["id"]]
        task, code = str(m["task_id"]), m["code"]
        if (
            r["status"] != "ok"
            or r["arm"] != name
            or any(
                r.get(k) != m[k] for k in ("task_id", "code", "text", "split", "set")
            )
        ):
            raise ValueError("Step 1 missing or mismatched baseline")
        original = pairs[task][code]["prompt"]
        rendered = join_prompt(
            instruction(cfg, arm["instruction"]), split_prompt(v["prompt"]).trajectory
        )
        if original != rendered:
            raise ValueError("Step 1 no-text surface differs from original")
        baseline.append({**r, "id": f"{task}:{code}", "prompt_sha256": sha(rendered)})
    identity = {
        "monitor": monitor,
        "instruction": arm["instruction"],
        "backend": BACKEND,
        "instruction_sha256": sha(instruction(cfg, arm["instruction"])),
        "runtime_receipt_sha256": file_sha256(runtime_path),
        "canary_receipt_sha256": file_sha256(canary_path),
        "scorer": scorer,
    }
    dep = {
        "status": "ready",
        "task_ids": tasks,
        "calibration_task_ids": selection["calibration"],
        "monitor_identity": identity,
        "baseline_scores": baseline,
        "runtime_receipt": runtime,
        "canary_receipt": canary,
        "provenance": {
            "step1": str(step1.relative_to(ROOT)),
            "source_commit": "76a112c",
            "files": {
                str(p.relative_to(ROOT)): file_sha256(p)
                for p in [
                    step1 / "selection.json",
                    step1 / "variants.jsonl",
                    step1 / "variants.manifest.json",
                    score_path,
                    runtime_path,
                    canary_path,
                ]
            },
        },
    }
    select(dep, pairs, cfg)
    return dep


def verify_dependency(dep: dict, cfg: dict) -> None:
    monitor = dep["monitor_identity"]["monitor"]
    spec = cfg["dependencies"][monitor]
    frozen = json.loads((ROOT / spec["path"]).read_text())
    if digest(dep) != spec["sha256"] or dep != frozen:
        raise ValueError("frozen scoring dependency changed")
    identity = dep["monitor_identity"]
    for label in ("runtime", "canary"):
        matches = [
            p
            for p in dep["provenance"]["files"]
            if p.endswith(f".heldout.{label}.json")
        ]
        if len(matches) != 1:
            raise ValueError("missing original Step 1 receipt")
        path = ROOT / matches[0]
        if (
            file_sha256(path) != identity[f"{label}_receipt_sha256"]
            or json.loads(path.read_text()) != dep[f"{label}_receipt"]
        ):
            raise ValueError("original Step 1 receipt mismatch")
    rows = dep["canary_receipt"]["rows"]
    keys = [(r["task_id"], r["code"]) for r in rows]
    if (
        len(keys) != 8
        or len(set(keys)) != 8
        or len({t for t, c in keys}) != 4
        or any(
            {c for t, c in keys if t == task} != {"honest", "backdoored"}
            for task, c in keys
        )
    ):
        raise ValueError("need exact eight unique Step 1 canary pairs")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--step1", type=Path, default=ROOT / config()["step1"])
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    cfg = config()
    for monitor in cfg["monitor_contracts"]["search_arms"]:
        dep = build(args.step1, monitor, cfg)
        fixed(args.out / f"{monitor}.json", dep)
        selected = select(
            dep,
            instructed_pairs(benchmark_pairs(cfg), arm_config(cfg, monitor)),
            arm_config(cfg, monitor),
        )
        print(
            json.dumps(
                {
                    "monitor": monitor,
                    "cohort": len(selected["task_ids"]),
                    "calibration": selected["calibration"],
                    "excluded": selected["excluded"],
                }
            )
        )


if __name__ == "__main__":
    main()
