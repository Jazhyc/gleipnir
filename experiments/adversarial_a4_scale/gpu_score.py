"""Turn exported Gleipnir jobs into receipts on the existing VM. Never starts it."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from experiments.adversarial_crossed_text.campaign import Campaign
from gleipnir.adversarial.contract import join_prompt, split_prompt
from .common import append, digest, fixed, lock, probability, read, sha, unique
from .dependency import arm_config, instruction, instructed_pairs
from .pipeline import check_manifest, require_live, sealed, validate_scores
from .selection import benchmark_pairs


def finite_result(result: dict) -> dict:
    probability(result["score"])
    if not all(math.isfinite(result[k]) for k in ("logit_0", "logit_1")):
        raise ValueError("nonfinite logits")
    if not isinstance(result["tokens"], int) or result["tokens"] < 1:
        raise ValueError("missing prompt token count")
    return {**result, "margin": result["logit_1"] - result["logit_0"]}


def check_runtime(actual: dict, expected: dict) -> None:
    if actual != expected:
        raise ValueError("runtime differs from Step 1 receipt; no backend substitution")


def run(batch: dict, dependency: dict, manifest: dict, out: Path, receipt_path: Path,
        *, scorer_factory=None, campaign=None) -> dict:
    """Fail closed on source/runtime drift; durable per-job journal supports resume."""
    cfg = manifest["config"]
    identity = batch["monitor_identity"]
    if identity != dependency["monitor_identity"]:
        raise ValueError("batch and Step 1 monitor identity differ")
    jobs = unique(batch["jobs"])
    if any(j["prompt_sha256"] != sha(j["prompt"]) for j in jobs.values()):
        raise ValueError("job prompt hash mismatch")
    campaign = campaign or Campaign(cfg, out)
    # This must precede imports that can load or initialize the GPU.
    campaign.check_deadline()
    if scorer_factory is None:
        from gleipnir.adversarial.scorer import Scorer
        scorer_factory = Scorer
    scorer = scorer_factory(identity["monitor"], "cuda", "bfloat16")
    runtime = scorer.identity()
    check_runtime(runtime, identity["scorer"])
    target_cfg = arm_config(cfg, identity["monitor"])
    pairs = instructed_pairs(benchmark_pairs(cfg), target_cfg)
    canary_rows = []
    for row in dependency["canary_receipt"]["rows"]:
        campaign.check_deadline()
        prompt = pairs[row["task_id"]][row["code"]]["prompt"]
        result = finite_result(scorer.score(prompt))
        base = None
        if runtime["adapter"] is not None:
            campaign.check_deadline()
            base = finite_result(scorer.score(prompt, adapter=False))
        canary_rows.append({"task_id": row["task_id"], "code": row["code"],
                            "prompt_sha256": sha(prompt), **result,
                            "reference_score": row["score"],
                            "adapter_off_score": base["score"] if base else None})
    passed = len(canary_rows) == 8 and all(abs(r["score"] - r["reference_score"]) <= 0.02 for r in canary_rows)
    if runtime["adapter"] is not None:
        passed = passed and max(abs(r["score"] - r["adapter_off_score"]) for r in canary_rows) > 1e-4
    check = {"passed": passed, "runtime": runtime, "batch_sha256": digest(batch), "rows": canary_rows}
    fixed(receipt_path.with_name(receipt_path.name + f".canary.{time.time_ns()}.json"), check)
    if not passed:
        raise RuntimeError("canary failed; no bulk jobs scored")
    fixed(receipt_path.with_suffix(".binding.json"), {"batch_sha256": digest(batch), "runtime": runtime,
                                                   "monitor_identity": identity})
    journal = receipt_path.with_suffix(".rows.jsonl")
    done = unique(read(journal))
    if not done.keys() <= jobs.keys():
        raise ValueError("unexpected score journal rows")
    for cid, row in done.items():
        if row["prompt_sha256"] != jobs[cid]["prompt_sha256"]:
            raise ValueError("stale scoring journal")
    started, tokens = time.monotonic(), 0
    from gleipnir.adversarial.scorer import TooLong
    for cid, job in jobs.items():
        if cid in done:
            continue
        campaign.check_deadline()
        try:
            result = {**finite_result(scorer.score(job["prompt"])), "status": "ok"}
            tokens += result["tokens"]
        except TooLong:
            result = {"status": "too_long"}
        except Exception as error:
            result = {"status": "infrastructure_failure", "error_type": type(error).__name__}
        row = {"id": cid, "prompt_sha256": job["prompt_sha256"], **result}
        append(journal, row)
        done[cid] = row
        if len(done) % 25 == 0 or len(done) == len(jobs):
            print(json.dumps({"completed": len(done), "jobs": len(jobs),
                              "prompt_tokens_per_second": tokens / max(time.monotonic() - started, 1)}), flush=True)
    receipt = {"batch_sha256": digest(batch), "monitor_identity": identity,
               "canary_passed": True, "canary": check, "runtime": runtime,
               "rows": [done[cid] for cid in jobs]}
    validate_scores(batch, receipt)
    fixed(receipt_path, receipt)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--jobs", type=Path, required=True)
    p.add_argument("--dependency", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads((args.out / "manifest.json").read_text())
    cfg = manifest["config"]
    require_live(manifest)
    check_manifest(manifest, instructed_pairs(benchmark_pairs(cfg), cfg))
    sealed(args.out, manifest)
    from .study import validate_dual_seal
    validate_dual_seal(args.out.parent)
    batch = json.loads(args.jobs.read_text())
    if batch.get("manifest") != digest(manifest):
        raise ValueError("scoring batch belongs to another manifest")
    dependency = json.loads(args.dependency.read_text())
    with lock(args.out.parent / "gpu-session"):
        run(batch, dependency, manifest, args.out, args.receipt)


if __name__ == "__main__":
    main()
