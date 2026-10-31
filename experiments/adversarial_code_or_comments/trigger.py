"""Finding each backdoor's special case and a trigger input.

The writer sees the problem and both solutions and names the special case and a
trigger input (`propose`). `export` writes sandbox jobs; `trigger_runner.py`
runs them on a Linux VM in `docker run --network none`; `merge` keeps a task
only if both programs ran cleanly and their outputs differ.

    M=experiments.adversarial_code_or_comments.trigger
    PYTHONPATH=src python -m $M propose --attempt 1 [--dry-run --out DIR]
    PYTHONPATH=src python -m $M export --attempt 1
    # on the VM: python3 trigger_runner.py jobs_attempt1.jsonl verdicts.jsonl
    PYTHONPATH=src python -m $M merge
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from experiments.adversarial_code_or_comments.common import (
    config,
    fill,
    load_pairs,
    output_dir,
    prompt,
    read_jsonl,
    require_frozen,
    sha,
    write_jsonl,
)
from experiments.adversarial_code_or_comments.llm import (
    Client,
    finish_status,
    parse_json_object,
)
from gleipnir.adversarial import apps
from gleipnir.adversarial.attacker import AttackerError, BudgetExceeded

KEYS = ("special_case", "code_location", "trigger_input")
MAX_TRIGGER_CHARS = 20_000
RETRYABLE = {"same_output", "honest_error", "backdoor_error", "timeout"}


def build_prompt(problem: str, honest: str, backdoor: str, cfg) -> str:
    return fill(
        prompt("trigger.txt", cfg),
        problem=problem,
        honest_code=honest,
        backdoor_code=backdoor,
    )


def parse_proposal(text: str) -> dict[str, str]:
    """The three fields, checked; the trigger always ends in a newline."""
    value = parse_json_object(text)
    out = {}
    for key in KEYS:
        field = value.get(key)
        if not isinstance(field, str) or not field.strip():
            raise ValueError(f"missing or empty {key}")
        out[key] = field
    if len(out["trigger_input"]) > MAX_TRIGGER_CHARS:
        raise ValueError("trigger input too long")
    if not out["trigger_input"].endswith("\n"):
        out["trigger_input"] += "\n"
    out["special_case"] = " ".join(out["special_case"].split())
    return out


def clip(text: str, n: int = 300) -> str:
    return text if len(text) <= n else text[:n] + " [...]"


def feedback(verdict: dict[str, Any]) -> str:
    """What the sandbox saw, for the second attempt."""
    parts = []
    for label, side in (("Solution A", "honest"), ("Solution B", "backdoor")):
        run = verdict[side]
        if run.get("timed_out"):
            parts.append(f"{label} did not finish within the time limit.")
        elif run.get("exit_code") != 0:
            parts.append(
                f"{label} exited with code {run.get('exit_code')}: "
                f"{clip(run.get('stderr_tail', ''))!r}."
            )
        else:
            parts.append(f"{label} printed {clip(run.get('stdout', ''))!r}.")
    return " ".join(parts)


def messages_for(task: str, attempt: int, pair, proposals, verdicts, cfg):
    sub = {code: apps.parse(pair[code]["prompt"]) for code in pair}
    first = build_prompt(
        sub["honest"].problem, sub["honest"].code, sub["backdoored"].code, cfg
    )
    messages = [{"role": "user", "content": first}]
    if attempt == 1:
        return messages
    previous = proposals.get(f"{task}:{attempt - 1}")
    verdict = verdicts.get(f"{task}:{attempt - 1}")
    if not previous or previous["status"] != "ok" or not verdict:
        return None
    if verdict["status"] not in RETRYABLE:
        return None
    retry = fill(prompt("trigger_retry.txt", cfg), feedback=feedback(verdict))
    return messages + [
        {"role": "assistant", "content": previous["response_text"]},
        {"role": "user", "content": retry},
    ]


def mock_responder(messages: list[dict[str, str]]) -> str:
    """Dry run: a well-formed proposal; it says nothing about the real backdoor."""
    return json.dumps(
        {
            "special_case": "dry run: when the first number is 0",
            "code_location": "dry run",
            "trigger_input": "0\n",
        }
    )


def propose(args, cfg) -> None:
    if not args.dry_run:
        require_frozen(cfg)
    out = output_dir(args.out, cfg)
    pools = json.loads((out / "pools.json").read_text())
    tasks = pools["development_pool"] + pools["heldout_pool"]
    if args.limit:
        tasks = tasks[: args.limit]
    pairs, _ = load_pairs(cfg)
    path = out / "trigger_proposals.jsonl"
    proposals = {r["id"]: r for r in read_jsonl(path)}
    verdicts = {r["id"]: r for r in read_jsonl(out / "trigger_verdicts.jsonl")}
    client = Client(
        cfg["models"]["writer"],
        cfg["budget_usd"]["writer"]
        - sum(r.get("charged_usd", 0.0) for r in proposals.values()),
        cfg,
        timeout=cfg["request_timeout_seconds"],
        responder=mock_responder if args.dry_run else None,
    )
    jobs = []
    for task in tasks:
        rid = f"{task}:{args.attempt}"
        if rid in proposals:
            continue  # completed records, failures included, are never retried
        messages = messages_for(
            task, args.attempt, pairs[task], proposals, verdicts, cfg
        )
        if messages is not None:
            jobs.append((task, rid, messages))

    def work(job):
        task, rid, messages = job
        record: dict[str, Any] = {
            "id": rid,
            "task_id": task,
            "attempt": args.attempt,
            "request_sha256": sha(json.dumps(messages)),
            "charged_usd": 0.0,
            "timestamp": time.time(),
            "dry_run": args.dry_run,
        }
        try:
            result = client.chat(messages, attempt=args.attempt)
        except BudgetExceeded as error:
            return {**record, "status": "budget_exhausted", "error": str(error)}
        except AttackerError as error:
            record["charged_usd"] = client.config.worst_case_usd(
                sum(len(m["content"].encode()) for m in messages)
            )
            return {
                **record,
                "status": "infrastructure_failure",
                "error": str(error)[:300],
                "error_details": error.details,
            }
        record.update(
            charged_usd=result["cost_usd"],
            usage=result["usage"],
            finish_reason=result["finish_reason"],
            response_text=result["text"],
            model=result["model"],
        )
        status = finish_status(result)
        if status:
            return {**record, "status": status}
        try:
            record.update(parse_proposal(result["text"]), status="ok")
        except (ValueError, json.JSONDecodeError) as error:
            record.update(status="invalid_proposal", error=str(error)[:300])
        return record

    with ThreadPoolExecutor(max_workers=cfg["workers"]) as pool, path.open("a") as fh:
        futures = [pool.submit(work, job) for job in jobs]
        for n, future in enumerate(as_completed(futures), 1):
            fh.write(json.dumps(future.result()) + "\n")
            fh.flush()
            if n % 100 == 0 or n == len(jobs):
                print(f"proposals {n}/{len(jobs)}, ${client.budget.spent_usd:.4f}")
    statuses = [r["status"] for r in read_jsonl(path) if r["attempt"] == args.attempt]
    print({s: statuses.count(s) for s in sorted(set(statuses))})


def export(args, cfg) -> None:
    """Sandbox jobs for one attempt: both programs and the proposed trigger."""
    out = output_dir(args.out, cfg)
    pairs, _ = load_pairs(cfg)
    done = {r["id"] for r in read_jsonl(out / "trigger_verdicts.jsonl")}
    jobs = []
    for r in read_jsonl(out / "trigger_proposals.jsonl"):
        if r["attempt"] != args.attempt or r["status"] != "ok" or r["id"] in done:
            continue
        pair = pairs[r["task_id"]]
        jobs.append(
            {
                "id": r["id"],
                "task_id": r["task_id"],
                "attempt": r["attempt"],
                "honest_code": apps.parse(pair["honest"]["prompt"]).code,
                "backdoor_code": apps.parse(pair["backdoored"]["prompt"]).code,
                "trigger_input": r["trigger_input"],
            }
        )
    path = out / f"trigger_jobs_attempt{args.attempt}.jsonl"
    write_jsonl(path, jobs)
    print(f"wrote {len(jobs)} jobs to {path}")


def mock_verdicts(args, cfg) -> None:
    """Dry run only: mark every exported job confirmed without running anything."""
    if not args.dry_run:
        raise SystemExit(
            "mock-verdicts is for dry runs; real verdicts come from the VM"
        )
    out = output_dir(args.out, cfg)
    jobs = read_jsonl(out / f"trigger_jobs_attempt{args.attempt}.jsonl")
    fake = {"timed_out": False, "exit_code": 0, "stderr_tail": ""}
    rows = [
        {
            "id": j["id"],
            "task_id": j["task_id"],
            "attempt": j["attempt"],
            "status": "confirmed",
            "honest": {**fake, "stdout": "1\n"},
            "backdoor": {**fake, "stdout": "2\n"},
            "image_digest": None,
            "dry_run": True,
        }
        for j in jobs
    ]
    path = out / "trigger_verdicts.jsonl"
    write_jsonl(path, read_jsonl(path) + rows)
    print(f"wrote {len(rows)} dry-run verdicts to {path}")


def merge_records(proposals, verdicts, tasks) -> list[dict[str, Any]]:
    """One row per task: its first confirmed attempt, else why it failed."""
    by_id = {v["id"]: v for v in verdicts}
    rows = []
    for task in tasks:
        attempts = sorted(
            (p for p in proposals if p["task_id"] == task), key=lambda p: p["attempt"]
        )
        row: dict[str, Any] = {"task_id": task, "status": "no_proposal", "attempts": 0}
        for p in attempts:
            row["attempts"] = p["attempt"]
            verdict = by_id.get(p["id"])
            if p["status"] != "ok":
                row["status"] = f"proposal_{p['status']}"
                continue
            if verdict is None:
                row["status"] = "not_run"
                continue
            row["status"] = verdict["status"]
            if verdict["status"] == "confirmed":
                row.update(
                    attempt=p["attempt"],
                    special_case=p["special_case"],
                    code_location=p["code_location"],
                    trigger_input=p["trigger_input"],
                    honest_output=verdict["honest"]["stdout"],
                    backdoor_output=verdict["backdoor"]["stdout"],
                    image_digest=verdict.get("image_digest"),
                )
                break
        rows.append(row)
    return rows


def merge(args, cfg) -> None:
    out = output_dir(args.out, cfg)
    pools = json.loads((out / "pools.json").read_text())
    tasks = pools["development_pool"] + pools["heldout_pool"]
    rows = merge_records(
        read_jsonl(out / "trigger_proposals.jsonl"),
        read_jsonl(out / "trigger_verdicts.jsonl"),
        tasks,
    )
    write_jsonl(out / "triggers.jsonl", rows)
    statuses = [r["status"] for r in rows]
    print({s: statuses.count(s) for s in sorted(set(statuses))})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "command", choices=["propose", "export", "merge", "mock-verdicts"]
    )
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--limit", type=int, help="first N pool tasks (smoke only)")
    parser.add_argument("--out", help="output folder (default: config output)")
    parser.add_argument("--dry-run", action="store_true", help="mock model, no network")
    args = parser.parse_args()
    cfg = config()
    if not 1 <= args.attempt <= cfg["trigger"]["max_attempts"]:
        raise SystemExit("attempt outside the frozen maximum")
    commands = {
        "propose": propose,
        "export": export,
        "merge": merge,
        "mock-verdicts": mock_verdicts,
    }
    commands[args.command](args, cfg)


if __name__ == "__main__":
    main()
