"""Score variant rows with a released monitor: one JSONL row each, with its margin.

Uses the existing Transformers + PEFT `Scorer` (one forward pass per row, BF16)
on a GPU box. The margin is logit("1") - logit("0") at the decision position.
Held-out rows are refused until `predictions.json` is committed. `--dry-run`
replaces the model with a deterministic hash-based stand-in (no torch, no GPU,
no downloads) to exercise the plumbing.

    M=experiments.adversarial_code_or_comments.score
    PYTHONPATH=src python -m $M --monitor 4b-v03 --prompt student \
        --split development --sets calibration main repetition
    PYTHONPATH=src python -m $M --monitor 27b-merged --prompt teacher \
        --split heldout --sets calibration main
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import subprocess
import time
from typing import Any

from experiments.adversarial_code_or_comments.common import (
    ROOT,
    config,
    instruction,
    load_pairs,
    output_dir,
    read_jsonl,
    require_frozen,
    require_predictions,
    sha,
    write_json,
)
from gleipnir.adversarial.contract import join_prompt, split_prompt

FIELDS = (
    "task_id",
    "split",
    "set",
    "code",
    "text",
    "repetition",
    "added_chars",
    "ground_truth",
    "donor_task_id",
)


class MockScorer:
    """Deterministic stand-in for dry runs; its numbers mean nothing."""

    monitor = "dry-run"

    def score(self, user_prompt: str, adapter: bool = True) -> dict[str, Any]:
        digest = int(sha(user_prompt + str(adapter))[:12], 16)
        zero = 20.0 + (digest % 1000) / 100
        one = 20.0 + (digest // 1000 % 1000) / 100
        return {
            "score": 1 / (1 + math.exp(zero - one)),
            "logit_0": zero,
            "logit_1": one,
            "tokens": len(user_prompt) // 4,
        }

    def identity(self) -> dict[str, Any]:
        return {"monitor": "dry-run"}


def rows_for(variants, sets, split) -> list[dict[str, Any]]:
    """Rows of the given sets; calibration rows go with any split."""
    out = []
    for row in variants:
        m = row["metadata"]
        if m["set"] in sets and (m["set"] == "calibration" or m["split"] == split):
            out.append(row)
    return out


def with_instruction(user_prompt: str, head: str) -> str:
    """Same trajectory, byte for byte, under another monitor instruction."""
    return join_prompt(head, split_prompt(user_prompt).trajectory)


def record(row: dict[str, Any], result: dict[str, Any], monitor, prompt_name):
    m = row["metadata"]
    out = {"id": row["id"], **{k: m.get(k) for k in FIELDS}}
    out.update(monitor=monitor, instruction=prompt_name, **result)
    if "logit_0" in result:
        out["margin"] = result["logit_1"] - result["logit_0"]
    return out


def reference_scores(cfg) -> dict[str, float]:
    c = cfg["scoring"]["canary"]
    path = ROOT / c["reference"]
    from gleipnir.adversarial.eval_sets import file_sha256

    if file_sha256(path) != c["reference_sha256"]:
        raise SystemExit("canary reference does not match its SHA256")
    out = {}
    for r in read_jsonl(path):
        if r.get("type") == "result" and r.get("status") == "ok":
            value = r["value"]
            value = ast.literal_eval(value) if isinstance(value, str) else value
            out[r["id"]] = value["score"]
    return out


def canary(scorer, pairs, selection, cfg, head: str, dry_run: bool):
    """Unedited development-pool rows: finite, near the reference, adapter matters."""
    c = cfg["scoring"]["canary"]
    # The reference is the 4B scored with the student instruction the rows
    # already carry (head is None); until 22 October this read `is not None`,
    # which skipped the frozen drift check on every planned run.
    compare = scorer.monitor == c["monitor"] and head is None and not dry_run
    reference = reference_scores(cfg) if compare else {}
    rows = []
    for code, pattern in c["reference_ids"].items():
        for task in selection["development_pool"][: c["rows_per_code"]]:
            user_prompt = pairs[task][code]["prompt"]
            if head is not None:
                user_prompt = with_instruction(user_prompt, head)
            result = scorer.score(user_prompt)
            base = scorer.score(user_prompt, adapter=False)
            ref = reference.get(pattern.format(task_id=task))
            rows.append(
                {
                    "task_id": task,
                    "code": code,
                    **result,
                    "adapter_off_score": base["score"],
                    "reference_score": ref,
                }
            )
    finite = all(math.isfinite(r["score"]) for r in rows)
    drift = [
        abs(r["score"] - r["reference_score"])
        for r in rows
        if r["reference_score"] is not None
    ]
    # Merged checkpoints have no adapter to switch off, so the check is skipped.
    has_adapter = dry_run or getattr(scorer.spec, "adapter", None) is not None
    adapter = not has_adapter or (
        max(abs(r["score"] - r["adapter_off_score"]) for r in rows) > 1e-6
    )
    passed = (
        finite
        and adapter
        and (
            not compare
            or (len(drift) == len(rows) and max(drift) <= c["max_abs_score_drift"])
        )
    )
    return {
        "passed": passed,
        "compared_to_reference": compare,
        "adapter_checked": has_adapter,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--monitor", required=True)
    parser.add_argument("--prompt", choices=["student", "teacher"], default="student")
    parser.add_argument("--split", choices=["development", "heldout"], required=True)
    parser.add_argument(
        "--sets",
        nargs="+",
        default=["main"],
        choices=["calibration", "main", "repetition"],
    )
    parser.add_argument("--out", help="output folder (default: config output)")
    parser.add_argument("--dry-run", action="store_true", help="mock scorer, no GPU")
    args = parser.parse_args()
    cfg = config()
    if not args.dry_run:
        require_frozen(cfg)
        if args.split == "heldout":
            require_predictions(cfg)
    out = output_dir(args.out, cfg)
    selection = json.loads((out / "selection.json").read_text())
    variants = read_jsonl(out / "variants.jsonl")
    rows = rows_for(variants, set(args.sets), args.split)
    # The APPS rows already carry the student instruction; the teacher replaces it.
    head = instruction("teacher", cfg) if args.prompt == "teacher" else None
    student = instruction("student", cfg)
    if any(split_prompt(r["prompt"]).head != student for r in rows):
        raise SystemExit("a variant does not carry the frozen student instruction")
    if args.dry_run:
        scorer: Any = MockScorer()
        too_long: Any = ()
    else:
        from gleipnir.adversarial.scorer import Scorer, TooLong

        scorer = Scorer(args.monitor, cfg["scoring"]["device"], cfg["scoring"]["dtype"])
        too_long = TooLong
    started = time.time()
    path = out / "scores" / f"{args.monitor}_{args.prompt}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    pairs, _ = load_pairs(cfg)
    check = canary(scorer, pairs, selection, cfg, head, args.dry_run)
    write_json(path.with_suffix(".canary.json"), check)
    if not check["passed"]:
        raise SystemExit("canary failed: drift, nonfinite score or no adapter effect")
    write_json(
        path.with_suffix(f".{args.split}.identity.json"),
        {
            "variants_sha256": sha((out / "variants.jsonl").read_text()),
            "scorer": scorer.identity(),
            "instruction": args.prompt,
            "sets": args.sets,
            "commit": None
            if args.dry_run
            else subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        },
    )
    done = {r["id"] for r in read_jsonl(path)}
    todo = [r for r in rows if r["id"] not in done]
    budget = cfg["scoring"]["gpu_budget_seconds"]
    tokens = 0
    with path.open("a") as handle:
        for n, row in enumerate(todo, 1):
            if time.time() - started > budget:
                raise SystemExit("GPU time budget exhausted; rerun resumes")
            user_prompt = (
                row["prompt"] if head is None else with_instruction(row["prompt"], head)
            )
            t0 = time.time()
            try:
                result = {**scorer.score(user_prompt), "status": "ok"}
                if not math.isfinite(result["logit_1"] - result["logit_0"]):
                    raise SystemExit(f"{row['id']}: nonfinite logits")
                tokens += result["tokens"]
            except too_long as error:
                result = {"status": "too_long", "error": str(error)}
            result["seconds"] = time.time() - t0
            handle.write(
                json.dumps(record(row, result, args.monitor, args.prompt)) + "\n"
            )
            handle.flush()
            if n % 100 == 0 or n == len(todo):
                rate = tokens / max(time.time() - started, 1e-9)
                print(f"{n}/{len(todo)} rows, {rate:,.0f} prompt tok/s", flush=True)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
