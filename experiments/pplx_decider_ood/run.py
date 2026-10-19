"""Score frozen Jev Noul states with a local pplx-decider checkpoint.

Runs in the isolated pplx-decider runtime (``/root/pplx-venv``), so it imports
only the standard library, torch and ``pplx_decider``. Labels are never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "experiments/pplx_decider_ood/config.json"
OUT = ROOT / "results/pplx_decider_ood"


def stable_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_question(config: dict, variant: str = "jev") -> tuple[str, dict]:
    spec = config["question" if variant == "jev" else "teacher_question"]
    question = json.loads((ROOT / spec["path"]).read_text())["question"]
    if stable_sha256(question) != spec["question_sha256"]:
        raise ValueError("Noul question differs from the frozen contract")
    return spec["question_id"], question


def load_states(config: dict, phase: str, limit: int | None) -> list[dict]:
    spec = config["inputs"][phase]
    path = ROOT / spec["path"]
    if file_sha256(path) != spec["sha256"]:
        raise ValueError(f"{phase} states differ from the frozen input")
    rows = [json.loads(line) for line in path.open()]
    if len(rows) != spec["rows"]:
        raise ValueError(f"{phase} has {len(rows)} rows, expected {spec['rows']}")
    for row in rows:
        if hashlib.sha256(row["state"].encode()).hexdigest() != row["state_sha256"]:
            raise ValueError(f"state hash mismatch for {row['id']}")
    return rows[:limit] if limit else rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("flash", "decider"), required=True)
    parser.add_argument("--phase", choices=("canary", "full"), required=True)
    parser.add_argument("--question", choices=("jev", "teacher"), default="jev")
    parser.add_argument("--path", choices=("fast", "reference"), required=True)
    parser.add_argument(
        "--limit", type=int, help="score only the first N rows (smoke test)"
    )
    args = parser.parse_args()

    import torch
    from pplx_decider import Decider

    config = json.loads(CONFIG.read_text())
    model = config["models"][args.model]
    key, question = load_question(config, args.question)
    rows = load_states(config, args.phase, args.limit)
    prefix = "" if args.question == "jev" else f"{args.question}_"
    name = f"{prefix}{args.phase}_{args.path}" + (
        f"_limit{args.limit}" if args.limit else ""
    )
    out_dir = OUT / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / f"{name}.jsonl"
    done = (
        {json.loads(line)["id"] for line in scores_path.open()}
        if scores_path.exists()
        else set()
    )
    todo = [row for row in rows if row["id"] not in done]
    print(f"{args.model} {name}: {len(done)} done, {len(todo)} to score", flush=True)

    checkpoint = Path(model["local_dir"])
    started = time.time()
    decider = Decider.from_pretrained(
        checkpoint, device="cuda", fast=args.path == "fast"
    )
    if args.path == "fast":
        decider.warmup()
    ready = time.time()
    temperature = decider.temperature
    chunk = config["runtime"]["request_chunk"] if args.path == "fast" else 1
    scored_tokens = 0
    scoring_seconds = 0.0
    with scores_path.open("a") as handle:
        for start in range(0, len(todo), chunk):
            batch = todo[start : start + chunk]
            requests = [
                {"state": row["state"], "questions": {key: question}} for row in batch
            ]
            tick = time.time()
            outputs = list(
                decider._run(requests)
            )  # (request, raw logits, encoded chunks)
            torch.cuda.synchronize()
            scoring_seconds += time.time() - tick
            for row, (_, logits, chunks) in zip(batch, outputs, strict=True):
                false_logit, true_logit = (float(v) for v in logits[key])
                if not (math.isfinite(false_logit) and math.isfinite(true_logit)):
                    raise ValueError(f"nonfinite logits for {row['id']}")
                margin = true_logit - false_logit
                tokens = sum(c.length for c in chunks)
                scored_tokens += tokens
                record = {
                    "id": row["id"],
                    "state_sha256": row["state_sha256"],
                    "logits": {"false": false_logit, "true": true_logit},
                    "margin": margin,
                    "noul": 1.0 / (1.0 + math.exp(-margin / temperature)),
                    "input_tokens": tokens,
                    "sequences": len(chunks),
                }
                handle.write(json.dumps(record) + "\n")
            handle.flush()
            rate = scored_tokens / scoring_seconds
            done_rows = start + len(batch)
            print(f"scored {done_rows}/{len(todo)} tok/s={rate:.0f}", flush=True)

    receipt = {
        "model": args.model,
        "repo": model["repo"],
        "revision": model["revision"],
        "phase": args.phase,
        "path": args.path,
        "rows_scored_this_run": len(todo),
        "temperature": temperature,
        "question": args.question,
        "question_sha256": stable_sha256(question),
        "input_sha256": config["inputs"][args.phase]["sha256"],
        "scores_sha256": file_sha256(scores_path),
        "load_and_warmup_seconds": ready - started,
        "scoring_seconds": scoring_seconds,
        "scored_input_tokens": scored_tokens,
        "prompt_tokens_per_second": scored_tokens / scoring_seconds
        if scoring_seconds
        else None,
        "requests_per_second": len(todo) / scoring_seconds if scoring_seconds else None,
        "gpu": torch.cuda.get_device_name(),
        "gpu_uuid": os.popen("nvidia-smi --query-gpu=uuid --format=csv,noheader")
        .read()
        .strip(),
        "torch": torch.__version__,
        "finished": time.time(),
    }
    (out_dir / f"{name}.receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("RUN_DONE", json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
