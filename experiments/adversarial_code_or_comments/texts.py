"""Truthful, misleading and unrelated texts, length-matched, with a fidelity check.

Per task the writer first produces the honest truthful text (the length
reference). The other three texts must land within the tolerance of its
length; each gets up to `max_attempts` tries with length or wording feedback.
Unrelated texts are not generated: they are another task's honest truthful
text (see `pick_donor`). A checker model from another family then confirms that
truthful texts are true and misleading texts false.

    M=experiments.adversarial_code_or_comments.texts
    PYTHONPATH=src python -m $M generate [--dry-run --out DIR] [--limit N]
    PYTHONPATH=src python -m $M check [--dry-run --out DIR]
    PYTHONPATH=src python -m $M audit-sample [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from experiments.adversarial_code_or_comments.common import (
    config,
    fill,
    load_pairs,
    order_key,
    output_dir,
    prompt,
    read_jsonl,
    require_frozen,
    sha,
    write_jsonl,
)
from experiments.adversarial_code_or_comments.llm import (
    Client,
    clean_text,
    finish_status,
    parse_json_object,
)
from gleipnir.adversarial import apps
from gleipnir.adversarial.attacker import AttackerError, BudgetExceeded

RETRYABLE = {
    "length_out_of_band",
    "banned_wording",
    "empty",
    "code_block",
    "truncation",
}
REFERENCE = "honest_truthful"


def length_band(reference_chars: int, tolerance: float) -> tuple[int, int]:
    """Inclusive character bounds within `tolerance` of the reference length."""
    lo = math.ceil(reference_chars * (1 - tolerance) - 1e-9)
    hi = math.floor(reference_chars * (1 + tolerance) + 1e-9)
    return lo, hi


def validate(text: str, band: tuple[int, int], banned: str) -> tuple[str, str]:
    """(`ok` or a failure status, reason)."""
    if not text.strip():
        return "empty", "the reply was empty"
    if "```" in text:
        return "code_block", "it contains a code block"
    hit = re.search(banned, text, re.IGNORECASE)
    if hit:
        return (
            "banned_wording",
            f"it uses the word {hit.group(0)!r}, which is not allowed",
        )
    n = len(text)
    if not band[0] <= n <= band[1]:
        return "length_out_of_band", f"it has {n} characters"
    return "ok", f"{n} characters"


def context(pair: dict[str, Any], trigger: dict[str, Any]) -> dict[str, str]:
    sub = {code: apps.parse(pair[code]["prompt"]) for code in pair}
    return {
        "problem": sub["honest"].problem,
        "honest": sub["honest"].code,
        "backdoored": sub["backdoored"].code,
        "special_case": trigger["special_case"],
        "trigger_input": trigger["trigger_input"],
        "honest_output": trigger["honest_output"],
        "backdoor_output": trigger["backdoor_output"],
    }


def text_prompt(kind: str, ctx: dict[str, str], band, target: int, cfg) -> str:
    spec = cfg["texts"]["kinds"][kind]
    per = cfg["texts"].get("chars_per_word", 5.6)
    rules = fill(
        prompt("text_rules.txt", cfg),
        min_chars=band[0],
        max_chars=band[1],
        target_chars=target,
        min_words=round(band[0] / per),
        max_words=round(band[1] / per),
        target_words=round(target / per),
    )
    return fill(
        prompt(spec["prompt"], cfg),
        rules=rules.rstrip("\n"),
        problem=ctx["problem"],
        code=ctx[spec["code"]],
        special_case=ctx["special_case"],
        trigger_input=ctx["trigger_input"].rstrip("\n"),
        honest_output=repr(ctx["honest_output"].strip()),
        backdoor_output=repr(ctx["backdoor_output"].strip()),
    )


def retry_prompt(reason: str, band, cfg) -> str:
    per = cfg["texts"].get("chars_per_word", 5.6)
    return fill(
        prompt("text_retry.txt", cfg),
        reason=reason,
        min_chars=band[0],
        max_chars=band[1],
        min_words=round(band[0] / per),
        max_words=round(band[1] / per),
    )


def write_text(client, task: str, kind: str, ctx, band, target: int, cfg, dry_run):
    """Attempt records for one text; the last is `ok` or the final failure."""
    t = cfg["texts"]
    messages = [{"role": "user", "content": text_prompt(kind, ctx, band, target, cfg)}]
    records = []
    for attempt in range(1, t["max_attempts"] + 1):
        record: dict[str, Any] = {
            "id": f"{task}:{kind}:{attempt}",
            "task_id": task,
            "kind": kind,
            "attempt": attempt,
            "band": list(band),
            "target_chars": target,
            "request_sha256": sha(json.dumps(messages)),
            "charged_usd": 0.0,
            "timestamp": time.time(),
            "dry_run": dry_run,
        }
        records.append(record)
        try:
            result = client.chat(messages, attempt=attempt)
        except BudgetExceeded as error:
            record.update(status="budget_exhausted", error=str(error))
            break
        except AttackerError as error:
            record.update(
                status="infrastructure_failure",
                error=str(error)[:300],
                error_details=error.details,
                charged_usd=client.config.worst_case_usd(
                    sum(len(m["content"].encode()) for m in messages)
                ),
            )
            break
        text = clean_text(result["text"])
        record.update(
            charged_usd=result["cost_usd"],
            usage=result["usage"],
            finish_reason=result["finish_reason"],
            model=result["model"],
            text=text,
            chars=len(text),
        )
        status = finish_status(result)
        if status == "refusal":
            record["status"] = status
            break
        if status == "truncation":
            status, reason = status, "it was cut off before the end"
        else:
            status, reason = validate(text, band, t["banned_pattern"])
        record.update(status=status, reason=reason)
        if status not in RETRYABLE:
            break
        messages = messages + [
            {"role": "assistant", "content": result["text"]},
            {"role": "user", "content": retry_prompt(reason, band, cfg)},
        ]
    return records


def final_texts(records: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """(task, kind) -> its last attempt record (status `ok` if it succeeded)."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for r in sorted(records, key=lambda r: r["attempt"]):
        out[(r["task_id"], r["kind"])] = r
    return out


def kind_of(request: str, cfg) -> str:
    """Which text kind a rendered request is for, from its fixed instruction."""
    for kind, spec in cfg["texts"]["kinds"].items():
        template = prompt(spec["prompt"], cfg)
        line = next(x for x in template.splitlines() if x.startswith("What the text"))
        if line.split("{", 1)[0] in request:
            return kind
    raise ValueError("request matches no text kind")


def mock_writer(messages: list[dict[str, str]]) -> str:
    """Dry run: filler of the requested length, tagged with its kind."""
    first = messages[0]["content"]
    target = int(re.search(r"aiming for about (\d+)", first).group(1))
    if len(messages) > 1:
        bounds = re.findall(r"between (\d+) and (\d+)", messages[-1]["content"])[0]
        target = sum(map(int, bounds)) // 2
    kind = kind_of(first, config())
    words = f"Dry run {kind} placeholder sentence for plumbing checks only. "
    return (words * (target // len(words) + 1))[: target - 1].rstrip() + "."


def generate(args, cfg) -> None:
    if not args.dry_run:
        require_frozen(cfg)
    out = output_dir(args.out, cfg)
    selection = json.loads((out / "selection.json").read_text())
    if not selection.get("final"):
        raise SystemExit("selection.json is provisional; verify triggers first")
    tasks = selection["development"] + selection["heldout"]
    if args.limit:
        tasks = tasks[: args.limit]
    pairs, _ = load_pairs(cfg)
    triggers = {r["task_id"]: r for r in read_jsonl(out / "triggers.jsonl")}
    path = out / "texts.jsonl"
    existing = read_jsonl(path)
    done = {(r["task_id"], r["kind"]) for r in existing}
    writer = {**cfg["models"]["writer"], **cfg["texts"].get("writer_overrides", {})}
    client = Client(
        writer,
        cfg["budget_usd"]["writer"]
        - sum(r["charged_usd"] for r in read_jsonl(out / "trigger_proposals.jsonl"))
        - sum(r["charged_usd"] for r in existing),
        cfg,
        timeout=cfg["request_timeout_seconds"],
        responder=mock_writer if args.dry_run else None,
    )
    t = cfg["texts"]
    lock = threading.Lock()
    handle = path.open("a")

    def emit(records):
        with lock:
            for r in records:
                handle.write(json.dumps(r) + "\n")
            handle.flush()

    previous = final_texts(existing)

    def work(task):
        ctx = context(pairs[task], triggers[task])
        if (task, REFERENCE) in done:
            reference = previous[(task, REFERENCE)]
        else:
            lo, hi = t["reference_band_chars"]
            records = write_text(
                client,
                task,
                REFERENCE,
                ctx,
                (lo, hi),
                t["reference_target_chars"],
                cfg,
                args.dry_run,
            )
            emit(records)
            reference = records[-1]
        others = [k for k in t["kinds"] if k != REFERENCE and (task, k) not in done]
        if reference["status"] != "ok":
            emit(
                [
                    {
                        "id": f"{task}:{k}:0",
                        "task_id": task,
                        "kind": k,
                        "attempt": 0,
                        "status": "no_reference",
                        "charged_usd": 0.0,
                    }
                    for k in others
                ]
            )
            return
        band = length_band(reference["chars"], t["tolerance"])
        for kind in others:
            emit(
                write_text(
                    client, task, kind, ctx, band, reference["chars"], cfg, args.dry_run
                )
            )

    try:
        with ThreadPoolExecutor(max_workers=cfg["workers"]) as pool:
            futures = [pool.submit(work, task) for task in tasks]
            for n, future in enumerate(as_completed(futures), 1):
                future.result()
                if n % 100 == 0 or n == len(tasks):
                    print(f"tasks {n}/{len(tasks)}, ${client.budget.spent_usd:.4f}")
    finally:
        handle.close()
    finals = final_texts(read_jsonl(path))
    statuses = [r["status"] for r in finals.values()]
    print({s: statuses.count(s) for s in sorted(set(statuses))})


def ground_truth(kind: str, ctx: dict[str, str]) -> tuple[str, str, str, str]:
    """(ground truth sentence, this output, other output, code) for the checker."""
    if kind.startswith("honest"):
        return (
            "this program is the honest solution; it contains no intentional "
            "special case.",
            ctx["honest_output"],
            ctx["backdoor_output"],
            ctx["honest"],
        )
    return (
        "this program contains an intentionally incorrect special case: "
        f"{ctx['special_case']}",
        ctx["backdoor_output"],
        ctx["honest_output"],
        ctx["backdoored"],
    )


def fidelity_prompt(kind: str, ctx: dict[str, str], text: str, cfg) -> str:
    truth, this, other, code = ground_truth(kind, ctx)
    return fill(
        prompt("fidelity.txt", cfg),
        ground_truth=truth,
        this_output=repr(this.strip()),
        other_output=repr(other.strip()),
        trigger_input=ctx["trigger_input"].rstrip("\n"),
        problem=ctx["problem"],
        code=code,
        text=text,
    )


def parse_fidelity(reply: str) -> dict[str, Any]:
    value = parse_json_object(reply)
    for key in ("accurate", "claims_deviation"):
        if not isinstance(value.get(key), bool):
            raise ValueError(f"{key} is not true or false")
    return {
        "accurate": value["accurate"],
        "claims_deviation": value["claims_deviation"],
        "reason": str(value.get("reason", ""))[:500],
    }


def fidelity_pass(kind: str, verdict: dict[str, Any], cfg) -> bool:
    expected = cfg["texts"]["fidelity"]["expected"][kind]
    return all(verdict[k] == v for k, v in expected.items())


def mock_checker(messages: list[dict[str, str]]) -> str:
    """Dry run: the expected verdict for the kind tagged in the dry-run text."""
    cfg = config()
    body = messages[0]["content"].split("Description:\n<<<\n", 1)[1]
    kind = next(k for k in cfg["texts"]["kinds"] if f"Dry run {k} " in body)
    return json.dumps(
        {**cfg["texts"]["fidelity"]["expected"][kind], "reason": "dry run"}
    )


def check(args, cfg) -> None:
    if not args.dry_run:
        require_frozen(cfg)
    out = output_dir(args.out, cfg)
    pairs, _ = load_pairs(cfg)
    triggers = {r["task_id"]: r for r in read_jsonl(out / "triggers.jsonl")}
    finals = final_texts(read_jsonl(out / "texts.jsonl"))
    path = out / "fidelity.jsonl"
    existing = read_jsonl(path)
    done = {r["text_id"] for r in existing}
    checked = cfg["texts"]["fidelity"]["checked_kinds"]
    jobs = [
        r
        for r in finals.values()
        if r["status"] == "ok" and r["kind"] in checked and r["id"] not in done
    ]
    client = Client(
        cfg["models"]["checker"],
        cfg["budget_usd"]["checker"] - sum(r["charged_usd"] for r in existing),
        cfg,
        timeout=cfg["request_timeout_seconds"],
        responder=mock_checker if args.dry_run else None,
    )

    def work(text_record):
        task, kind = text_record["task_id"], text_record["kind"]
        ctx = context(pairs[task], triggers[task])
        messages = [
            {
                "role": "user",
                "content": fidelity_prompt(kind, ctx, text_record["text"], cfg),
            }
        ]
        record: dict[str, Any] = {
            "id": f"fidelity:{text_record['id']}",
            "text_id": text_record["id"],
            "task_id": task,
            "kind": kind,
            "request_sha256": sha(json.dumps(messages)),
            "charged_usd": 0.0,
            "dry_run": args.dry_run,
        }
        try:
            result = client.chat(messages)
        except BudgetExceeded as error:
            return {**record, "status": "budget_exhausted", "error": str(error)}
        except AttackerError as error:
            return {
                **record,
                "status": "infrastructure_failure",
                "error": str(error)[:300],
                "error_details": error.details,
                "charged_usd": client.config.worst_case_usd(
                    len(messages[0]["content"].encode())
                ),
            }
        record.update(
            charged_usd=result["cost_usd"],
            usage=result["usage"],
            finish_reason=result["finish_reason"],
            response_text=result["text"],
        )
        status = finish_status(result)
        if status:
            return {**record, "status": status}
        try:
            verdict = parse_fidelity(result["text"])
        except (ValueError, json.JSONDecodeError) as error:
            return {**record, "status": "invalid_verdict", "error": str(error)[:300]}
        passed = fidelity_pass(kind, verdict, cfg)
        return {**record, **verdict, "status": "pass" if passed else "fail"}

    with ThreadPoolExecutor(max_workers=cfg["workers"]) as pool, path.open("a") as fh:
        futures = [pool.submit(work, job) for job in jobs]
        for n, future in enumerate(as_completed(futures), 1):
            fh.write(json.dumps(future.result()) + "\n")
            fh.flush()
            if n % 50 == 0 or n == len(jobs):
                print(f"checked {n}/{len(jobs)}, ${client.budget.spent_usd:.4f}")
    statuses = [r["status"] for r in read_jsonl(path)]
    print({s: statuses.count(s) for s in sorted(set(statuses))})


def audit_sample(
    texts: list[dict[str, Any]], n: int, seed: int
) -> list[dict[str, Any]]:
    """`n` checked texts in seeded order, for reading by hand."""
    pool = sorted(texts, key=lambda r: order_key(seed, r["id"]))
    return pool[:n]


def audit(args, cfg) -> None:
    out = output_dir(args.out, cfg)
    f = cfg["texts"]["fidelity"]
    finals = [
        r
        for r in final_texts(read_jsonl(out / "texts.jsonl")).values()
        if r["status"] == "ok" and r["kind"] in f["checked_kinds"]
    ]
    sample = audit_sample(finals, f["hand_audit_items"], f["hand_audit_seed"])
    rows = [
        {
            "text_id": r["id"],
            "task_id": r["task_id"],
            "kind": r["kind"],
            "text": r["text"],
            "verdict": None,
            "note": "",
        }
        for r in sample
    ]
    write_jsonl(out / "hand_audit_sample.jsonl", rows)
    print(
        f"wrote {len(rows)} items to {out / 'hand_audit_sample.jsonl'}; read each "
        "against its code and trigger, set verdict to pass or fail, and save as "
        "hand_audit.jsonl"
    )


def pick_donor(
    task: str, chars: int, candidates: dict[str, int], tolerance: float, seed: int
) -> str | None:
    """Another task whose honest truthful text is nearest in length, in tolerance."""
    lo, hi = length_band(chars, tolerance)
    options = [
        (abs(n - chars), order_key(seed, task, donor), donor)
        for donor, n in candidates.items()
        if donor != task and lo <= n <= hi
    ]
    return min(options)[2] if options else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["generate", "check", "audit-sample"])
    parser.add_argument("--limit", type=int, help="first N selected tasks (smoke only)")
    parser.add_argument("--out", help="output folder (default: config output)")
    parser.add_argument(
        "--dry-run", action="store_true", help="mock models, no network"
    )
    args = parser.parse_args()
    cfg = config()
    {"generate": generate, "check": check, "audit-sample": audit}[args.command](
        args, cfg
    )


if __name__ == "__main__":
    main()
