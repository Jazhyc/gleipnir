"""Pre-registered five-call development-only gates; never score their proposals."""
from __future__ import annotations

import argparse
import json
from collections import Counter

from .common import ROOT, config, digest, fixed, read, sha, source_identity
from .pipeline import generate
from .selection import benchmark_pairs


def development_tasks(cfg: dict) -> list[str]:
    selection = json.loads((ROOT / cfg["step1"] / "selection.json").read_text())
    tasks = sorted(selection["development"], key=lambda t: sha(f"smoke:{cfg['smoke']['seed']}:{t}"))[:5]
    if len(tasks) != 5 or set(tasks) & (set(selection["heldout"]) | set(selection["calibration"])):
        raise ValueError("smoke must use five distinct Step 1 development tasks")
    return tasks


def run(cfg: dict, out, pairs: dict) -> dict:
    tasks = development_tasks(cfg)
    fixed(out / "preregistration.json", {"smoke": cfg["smoke"], "tasks": tasks,
                                         "sources": source_identity(), "prompt": sha((__import__('pathlib').Path(__file__).parent / 'prompt.txt').read_text())})
    summaries = []
    for index, spec in enumerate(cfg["smoke"]["candidates"]):
        stage_cfg = json.loads(json.dumps(cfg))
        stage_cfg.update(writer=spec, attempts=1, curve_attempts=[1])
        manifest = {"stage": "smoke", "config": stage_cfg, "sources": source_identity(),
                    "selection": {"task_ids": tasks},
                    "pair_hashes": {t: {"backdoored": sha(pairs[t]["backdoored"]["prompt"])} for t in tasks}}
        dest = out / str(index)
        fixed(dest / "manifest.json", manifest)
        generate(dest, manifest, pairs)
        records = read(dest / "generations.jsonl")
        valid = sum(r["status"] == "ok" for r in records)
        summary = {"candidate": index, "writer": spec, "valid": valid, "calls": len(records),
                   "passed": len(records) == 5 and valid >= 4,
                   "statuses": dict(Counter(r["status"] for r in records)),
                   "charged_usd": sum(r["charged_usd"] for r in records),
                   "receipts_sha256": digest(records),
                   "mean_prompt_tokens": sum((r.get("response", {}).get("usage", {}).get("prompt_tokens") or 0) for r in records) / 5,
                   "mean_completion_tokens": sum((r.get("response", {}).get("usage", {}).get("completion_tokens") or 0) for r in records) / 5,
                   "usage_complete": all(all(r.get("response", {}).get("usage", {}).get(k) is not None for k in ("prompt_tokens", "completion_tokens")) for r in records)}
        fixed(dest / "summary.json", summary)
        summaries.append(summary)
        print(json.dumps({k:v for k,v in summary.items() if k != 'writer'}), flush=True)
        if summary["passed"]:
            result = {"status": "passed", "selected_writer": spec, "summaries": summaries,
                      "charged_usd": sum(s["charged_usd"] for s in summaries)}
            fixed(out / "gate.json", result)
            return result
    result = {"status": "failed", "summaries": summaries,
              "charged_usd": sum(s["charged_usd"] for s in summaries)}
    fixed(out / "gate.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.parse_args()
    cfg = config()
    result = run(cfg, ROOT / cfg["smoke"]["output"], benchmark_pairs(cfg))
    print(json.dumps({"gate": result["status"], "charged_usd": result["charged_usd"]}))


if __name__ == "__main__":
    main()
