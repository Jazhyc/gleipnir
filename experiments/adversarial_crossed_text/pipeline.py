"""Prepare, audit and cross identical documentation over each code pair."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from experiments.adversarial_code_or_comments.llm import (
    finish_status,
    parse_json_object,
)
from experiments.adversarial_code_or_comments.texts import (
    context,
    length_band,
    validate,
)
from experiments.adversarial_code_or_comments.trigger_runner import normalise
from experiments.adversarial_code_or_comments.variants import (
    check_variant,
    docstring,
    edited_prompt,
    insert_header,
)
from experiments.adversarial_crossed_text.common import (
    Calls,
    append,
    bind,
    config,
    digest,
    exclusive,
    frozen,
    load_pairs,
    order_key,
    output,
    prior,
    prompt,
    read_jsonl,
    seal_variants,
    sha,
    validate_import,
    validate_pipeline,
    write_json,
    write_jsonl,
)
from gleipnir.adversarial import apps


def select(cfg, old, triggers):
    used = set(old["development"] + old["heldout"])
    seed = cfg["selection"]["seed"]

    def sort(ids):
        return sorted(ids, key=lambda t: order_key(seed, t))

    fresh = sort({t for t, r in triggers.items() if r["status"] == "confirmed"} - used)
    if len(fresh) != cfg["selection"]["heldout_tasks"]:
        raise ValueError("fresh cohort size differs from protocol")
    dev = sort(old["development"])[: cfg["selection"]["development_tasks"]]
    cal = [t for t in old["development_pool"] if t not in set(dev + fresh)]
    assert not (set(dev) & set(fresh) or set(cal) & set(dev + fresh))
    return {
        "development": dev,
        "heldout": fresh,
        "calibration": cal,
        "development_pool": old["development_pool"],
        "prior_used_tasks": len(used),
        "novelty": "unused in prior comment study; prior no-comment scores may exist",
    }


def prepare(out, cfg):
    old, triggers = prior(cfg)
    pairs, _ = load_pairs(cfg)
    selection = select(cfg, old, triggers)
    if (
        not set(
            selection["development"] + selection["heldout"] + selection["calibration"]
        )
        <= pairs.keys()
    ):
        raise ValueError("selected task missing code pair")
    write_json(out / "selection.json", selection)
    print({k: len(selection[k]) for k in ("development", "heldout", "calibration")})


def completed_payloads(cfg, *, dry_run=False):
    """Reconstruct inherited bytes exclusively from the pinned source artifacts."""
    from experiments.adversarial_crossed_text.common import ROOT

    spec = cfg.get("amendment_2", cfg["amendment_1"])["import"]
    source = ROOT / spec["source"]
    for name, expected in spec["files"].items():
        if digest(source / name) != expected:
            raise ValueError(f"import source artifact changed: {name}")
    identity = json.loads((source / "pipeline.identity.json").read_text())
    if (
        identity["config_sha256"] != spec["config_sha256"]
        or identity["dry_run"] != dry_run
    ):
        raise ValueError("import source configuration or mock/live mode mismatch")
    for name in ("trigger_audit", "task_restatement", "text_rules", "text_check"):
        path = f"experiments/adversarial_crossed_text/prompts/{name}.txt"
        if identity["sources"][path] != cfg["prompts_sha256"][name]:
            raise ValueError(f"imported prompt changed: {name}")
    amendment2 = "amendment_2" in cfg
    keep = lambda kind: (
        kind != "cover_story" if amendment2 else kind == "task_restatement"
    )
    texts = [r for r in read_jsonl(source / "texts.jsonl") if keep(r["condition"])]
    checks = [
        r for r in read_jsonl(source / "text_checks.jsonl") if keep(r["condition"])
    ]
    calls = read_jsonl(source / "calls.jsonl")
    imported_ids = {
        r["request_id"]
        for r in calls
        if r["request_id"].startswith("trigger:")
        or any(
            f":{kind}:" in r["request_id"] for kind in cfg["conditions"] if keep(kind)
        )
    }
    latest = {r["request_id"]: r for r in calls if r["request_id"] in imported_ids}
    if any(r["status"] != "settled" or "result" not in r for r in latest.values()):
        raise ValueError("only completed calls may be imported")
    if any(r.get("dry_run") != dry_run for r in calls):
        raise ValueError("mixed imported call provenance")
    selection = json.loads((source / "selection.json").read_text())
    pairs, _ = load_pairs(cfg)
    _, triggers = prior(cfg)
    eligible(
        source,
        pairs,
        triggers,
        required=selection["development"] + selection["heldout"],
    )
    imported_texts = {(r["task_id"], r["condition"]): r for r in texts}
    cells = text_cells(selection, imported_texts, {}, cfg)
    reads = []
    for r in read_jsonl(source / "hand_audit.jsonl"):
        cell = cells.get((r["task_id"], r["condition"]))
        if cell and keep(r["condition"]) and (amendment2 or cell["record"]):
            expected = cell_identity(cell, pairs[r["task_id"]], triggers[r["task_id"]])
            if all(r.get(k) == v for k, v in expected.items()):
                hand_verdict(r, expected)
                reads.append(r)

    def encoded(rows):
        return "".join(json.dumps(r) + "\n" for r in rows).encode()

    return {
        "trigger_audit.jsonl": (source / "trigger_audit.jsonl").read_bytes(),
        "trigger_hand_reads.jsonl": (source / "trigger_hand_reads.jsonl").read_bytes(),
        "texts.jsonl": encoded(texts),
        "text_checks.jsonl": encoded(checks),
        "hand_audit.jsonl": encoded(reads),
        "calls.jsonl": encoded([r for r in calls if r["request_id"] in imported_ids]),
        "inherited_calls.jsonl": (
            (source / "inherited_calls.jsonl").read_bytes() if amendment2 else b""
        )
        + (source / "calls.jsonl").read_bytes(),
    }


def import_completed(out, cfg, *, dry_run=False):
    """Import pinned completed work, never rewrite bindings or ledger reservations."""
    from experiments.adversarial_crossed_text.common import ROOT

    spec = cfg.get("amendment_2", cfg["amendment_1"])["import"]
    if (ROOT / spec["source"]).resolve() == out.resolve():
        raise ValueError("import requires a new output directory")
    if (out / "import.manifest.json").exists():
        validate_import(out, cfg, dry_run)
        return
    if digest(out / "selection.json") != spec["files"]["selection.json"]:
        raise ValueError("import selection changed")
    payloads = completed_payloads(cfg, dry_run=dry_run)
    if any((out / n).exists() for n in payloads):
        raise SystemExit("import destination already contains work; use a fresh folder")
    for name, data in payloads.items():
        (out / name).write_bytes(data)
    manifest = {
        "source": spec,
        "dry_run": dry_run,
        "prefixes": {
            n: {"bytes": len(data), "sha256": digest(out / n)}
            for n, data in payloads.items()
        },
    }
    write_json(out / "import.manifest.json", manifest)
    validate_import(out, cfg, dry_run)
    print(
        {
            "imported_restatements": len(final_texts(out)),
            "reused_hand_verdicts": len(read_jsonl(out / "hand_audit.jsonl")),
        }
    )


def fill(template, values):
    # A single regex substitution cannot replace braces that came from values.
    return re.sub(r"\{([a-z_]+)\}", lambda m: str(values.get(m[1], m[0])), template)


def task_context(pair, trigger):
    return context(pair, trigger)


def input_identity(pair, trigger):
    """Bind verdicts to the trigger, problem and both submitted programs."""
    parsed = {code: apps.parse(pair[code]["prompt"]) for code in pair}
    if len({p.problem for p in parsed.values()}) != 1:
        raise ValueError("paired programs have different problem statements")
    return {
        "trigger_sha256": sha(json.dumps(trigger, sort_keys=True)),
        "problem_sha256": sha(parsed["honest"].problem),
        "code_pair_sha256": sha(
            json.dumps({c: p.code for c, p in parsed.items()}, sort_keys=True)
        ),
    }


def hand_verdict(record, expected):
    """An explicit disposition, including rejection, must name current inputs."""
    if (
        type(record.get("valid")) is not bool
        or not isinstance(record.get("reason"), str)
        or not record["reason"].strip()
        or any(k not in record or record[k] != v for k, v in expected.items())
    ):
        raise ValueError("hand verdict needs boolean valid, reason and current hashes")
    return record["valid"]


def mock_reply(messages):
    content = messages[-1]["content"]
    if "INPUT_AUDIT" in content:
        match = re.search(r"HONEST_OUTPUT_JSON=(.*)", content)
        return json.dumps(
            {
                "valid_input": True,
                "expected_output": json.loads(match[1]),
                "reason": "mock verdict",
            }
        )
    if "TEXT_CHECK" in content:
        return json.dumps({"matches_condition": True, "reason": "mock verdict"})
    match = re.search(r"TARGET_CHARS=(\d+)", content)
    n = int(match[1]) if match else 600
    words = (
        "This description discusses a short programming task "
        "with deterministic input and output. "
    )
    return (words * 20)[:n].rstrip()


def parse_result(result):
    if result["status"] != "ok":
        return result["status"], None
    status = finish_status(result)
    if status:
        return status, None
    try:
        return "ok", parse_json_object(result["text"])
    except (ValueError, TypeError):
        return "invalid_json", None


def run_audit(args, out, cfg):
    selection = json.loads((out / "selection.json").read_text())
    pairs, _ = load_pairs(cfg)
    _, triggers = prior(cfg)
    calls = Calls(out, cfg, "checker", args.dry_run, mock_reply)
    existing = {r["task_id"]: r for r in read_jsonl(out / "trigger_audit.jsonl")}
    tasks = selection["development"] + selection["heldout"]

    def work(task):
        identity = input_identity(pairs[task], triggers[task])
        if task in existing:
            if any(existing[task].get(k) != v for k, v in identity.items()):
                raise ValueError(
                    "trigger audit inputs changed; use a new output directory"
                )
            return
        ctx = task_context(pairs[task], triggers[task])
        # Do not reveal either output to the real checker. It solves the task.
        text = fill(prompt("trigger_audit", cfg), ctx)
        if args.dry_run:
            text += "\nHONEST_OUTPUT_JSON=" + json.dumps(ctx["honest_output"])
        messages = [{"role": "user", "content": text}]
        result = calls.chat(f"trigger:{task}", messages)
        status, verdict = parse_result(result)
        passed = bool(
            status == "ok"
            and verdict.get("valid_input") is True
            and isinstance(verdict.get("expected_output"), str)
            and normalise(verdict["expected_output"]) == normalise(ctx["honest_output"])
        )
        append(
            out / "trigger_audit.jsonl",
            {
                "task_id": task,
                "status": status,
                "passed": passed,
                "verdict": verdict,
                **identity,
            },
        )

    with ThreadPoolExecutor(6) as pool:
        list(pool.map(work, tasks[: args.limit or len(tasks)]))
    print(Counter(r["passed"] for r in read_jsonl(out / "trigger_audit.jsonl")))


def eligible(out, pairs=None, triggers=None, *, required=()):
    """Every rejected/ambiguous trigger requires a human/agent hand verdict."""
    audits = {r["task_id"]: r for r in read_jsonl(out / "trigger_audit.jsonl")}
    overrides = {r["task_id"]: r for r in read_jsonl(out / "trigger_hand_reads.jsonl")}
    if pairs is None or triggers is None:
        cfg = config()
        pairs, _ = load_pairs(cfg)
        _, triggers = prior(cfg)
    if set(required) - audits.keys():
        raise SystemExit("audit every selected trigger before building")
    accepted = set()
    for task, r in audits.items():
        identity = input_identity(pairs[task], triggers[task])
        if any(r.get(k) != v for k, v in identity.items()):
            raise ValueError("trigger audit has stale or missing input hashes")
        passed = r.get("passed") is True and r.get("status") == "ok"
        if task in overrides:
            passed = hand_verdict(overrides[task], {"task_id": task, **identity})
        elif task in required and not passed:
            raise SystemExit("read every rejected/ambiguous trigger first")
        if passed:
            accepted.add(task)
    return accepted


def text_key(record):
    return (record["task_id"], record["condition"], record["text_sha256"])


def final_texts(out):
    rows = {}
    for r in read_jsonl(out / "texts.jsonl"):
        if r.get("text_sha256") != sha(r["text"]):
            raise ValueError("text content does not match text_sha256")
        rows[(r["task_id"], r["condition"])] = r
    return rows


def cover_round_bound(out, cfg, tasks, pairs, triggers, texts, *, dry_run=False):
    """Bound all remaining requests before the round, with byte-based pricing."""
    from experiments.adversarial_code_or_comments.llm import endpoint
    from experiments.adversarial_crossed_text.common import spent
    from experiments.adversarial_crossed_text.campaign import Campaign

    amendment = cfg["amendment_2"]
    model = endpoint(cfg["models"][amendment["writer_model_role"]])
    histories = Counter(
        r["task_id"]
        for r in read_jsonl(out / "texts.jsonl")
        if r["condition"] == "cover_story"
    )
    worst = 0.0
    for task in tasks:
        previous = texts.get((task, "cover_story"))
        if previous and previous["status"] not in {
            "length_out_of_band",
            "banned_wording",
            "empty",
            "code_block",
            "truncation",
        }:
            continue
        reference = texts[(task, "task_restatement")]
        band = length_band(len(reference["text"]), cfg["texts"]["tolerance"])
        values = {
            **task_context(pairs[task], triggers[task]),
            "target_chars": len(reference["text"]),
            "min_chars": band[0],
            "max_chars": band[1],
        }
        text = (
            fill(prompt("cover_story", cfg), values)
            + "\n"
            + fill(prompt("text_rules", cfg), values)
        )
        remaining = max(0, amendment["max_attempts"] - histories[task])
        # Feedback is limited to 512 bytes; the prior reply is never resent.
        worst += remaining * model.worst_case_usd(len(text.encode()) + 512)
    campaign = Campaign(cfg, out, mock=dry_run)
    ledger = campaign.latest().values()
    campaign_used = sum(r["charge"] for r in ledger if r["api"] == "tinker")
    if (
        spent(out, "writer") + worst > cfg["budget_usd"]["writer"]
        or campaign_used + worst > cfg["campaign"]["caps_usd"]["tinker"]
    ):
        raise SystemExit("entire remaining cover round cannot fit worst-case caps")
    return worst


def generate(args, out, cfg):
    selection = json.loads((out / "selection.json").read_text())
    pairs, _ = load_pairs(cfg)
    _, triggers = prior(cfg)
    texts = final_texts(out)
    amendment = cfg.get("amendment_2")
    calls = Calls(
        out,
        cfg,
        "writer",
        args.dry_run,
        mock_reply,
        model_role=amendment["writer_model_role"] if amendment else None,
    )
    allowed = eligible(out, pairs, triggers)
    tasks = [t for t in selection["development"] + selection["heldout"] if t in allowed]

    if amendment and args.dry_run and not (out / "import.manifest.json").exists():
        # Fresh offline runs need synthetic stand-ins for the frozen imports.
        # They contain no paid replies and can never unlock live scoring.
        for task in tasks:
            for kind in (
                "task_restatement",
                "specific_accusation",
                "generic_accusation",
            ):
                if (task, kind) in texts:
                    continue
                reply = mock_reply(
                    [{"content": f"TARGET_CHARS={cfg['texts']['target']}"}]
                )
                record = {
                    "task_id": task,
                    "condition": kind,
                    "status": "ok",
                    "text": reply,
                    "text_sha256": sha(reply),
                    "dry_run": True,
                    "mock_import_fixture": True,
                }
                append(out / "texts.jsonl", record)
                texts[(task, kind)] = record
    if amendment:
        bound = cover_round_bound(
            out, cfg, tasks, pairs, triggers, texts, dry_run=args.dry_run
        )
        print({"remaining_cover_round_worst_case_usd": bound}, flush=True)

    def work(task):
        ctx = task_context(pairs[task], triggers[task])
        for kind in (
            "task_restatement",
            "specific_accusation",
            "cover_story",
            "generic_accusation",
        ):
            if amendment and kind not in amendment["regenerate_conditions"]:
                continue
            key = task, kind
            previous = texts.get(key)
            if previous and (
                previous["status"] == "ok"
                or previous["status"]
                not in {
                    "length_out_of_band",
                    "banned_wording",
                    "empty",
                    "code_block",
                    "truncation",
                }
            ):
                continue
            reference = texts.get((task, "task_restatement"))
            if kind != "task_restatement" and (
                not reference or reference["status"] != "ok"
            ):
                continue
            band = (
                tuple(cfg["texts"]["reference_band"])
                if kind == "task_restatement"
                else length_band(len(reference["text"]), cfg["texts"]["tolerance"])
            )
            target = (
                cfg["texts"]["target"]
                if kind == "task_restatement"
                else len(reference["text"])
            )
            values = {
                **ctx,
                "target_chars": target,
                "min_chars": band[0],
                "max_chars": band[1],
            }
            text = fill(prompt(kind, cfg), values)
            text += "\n" + fill(prompt("text_rules", cfg), values)
            messages = [{"role": "user", "content": text}]
            # Reconstruct retry feedback exactly when resuming a partial task.
            prior_rows = [
                r
                for r in read_jsonl(out / "texts.jsonl")
                if (r["task_id"], r["condition"]) == key
            ]
            for r in [] if amendment else prior_rows:
                messages.extend(
                    [
                        {"role": "assistant", "content": r.get("text", "")},
                        {"role": "user", "content": r["feedback"]},
                    ]
                )
            if amendment and prior_rows:
                messages.append({"role": "user", "content": prior_rows[-1]["feedback"]})
            attempts = (
                amendment["max_attempts"] if amendment else cfg["texts"]["max_attempts"]
            )
            for attempt in range(len(prior_rows) + 1, attempts + 1):
                result = calls.chat(f"text:{task}:{kind}:{attempt}", messages, attempt)
                reply = result.get("text", "").strip()
                status = (
                    result["status"]
                    if result["status"] != "ok"
                    else finish_status(result)
                )
                if status is None:
                    status, reason = validate(
                        reply, band, cfg["texts"]["banned_pattern"]
                    )
                else:
                    reason = status
                feedback = (
                    f"The previous reply failed because {reason}. "
                    f"Return {band[0]} to {band[1]} characters. "
                    f"TARGET_CHARS={target}. Return only prose."
                )
                if amendment and len(feedback.encode()) > 512:
                    raise ValueError(
                        "format feedback exceeds pre-registered byte bound"
                    )
                record = {
                    "task_id": task,
                    "condition": kind,
                    "attempt": attempt,
                    "status": status,
                    "text": reply,
                    "text_sha256": sha(reply),
                    "writer_reply_sha256": sha(result.get("text", "")),
                    "feedback": feedback,
                    "band": list(band),
                }
                append(out / "texts.jsonl", record)
                texts[key] = record
                if status not in {
                    "length_out_of_band",
                    "banned_wording",
                    "empty",
                    "code_block",
                    "truncation",
                }:
                    break
                if amendment:
                    messages = [
                        {"role": "user", "content": text},
                        {"role": "user", "content": feedback},
                    ]
                    continue
                messages.extend(
                    [
                        {"role": "assistant", "content": reply},
                        {"role": "user", "content": feedback},
                    ]
                )
        print(
            task,
            Counter(r["status"] for (t, _), r in texts.items() if t == task),
            flush=True,
        )

    with ThreadPoolExecutor(6) as pool:
        list(pool.map(work, tasks[: args.limit or len(tasks)]))


def checker_input(record, pairs, triggers, cfg):
    task = record["task_id"]
    values = {
        **task_context(pairs[task], triggers[task]),
        "condition": record["condition"],
        "text": record["text"],
    }
    messages = [{"role": "user", "content": fill(prompt("text_check", cfg), values)}]
    identity = sha(
        json.dumps(
            {
                "messages": messages,
                "inputs": input_identity(pairs[task], triggers[task]),
                "checker": cfg["models"]["checker"],
            },
            sort_keys=True,
        )
    )
    return messages, identity


def current_checks(out, texts, pairs, triggers, cfg):
    """Old verdicts never authorize text checked against different context."""
    stored = {
        (*text_key(r), r.get("checker_input_sha256")): r
        for r in read_jsonl(out / "text_checks.jsonl")
    }
    checks = {}
    for r in texts.values():
        if r["status"] == "ok":
            _, identity = checker_input(r, pairs, triggers, cfg)
            key = text_key(r)
            if (*key, identity) in stored:
                checks[key] = stored[(*key, identity)]
    return checks


def check(args, out, cfg):
    pairs, _ = load_pairs(cfg)
    _, triggers = prior(cfg)
    calls = Calls(out, cfg, "checker", args.dry_run, mock_reply)
    texts = final_texts(out)
    done = current_checks(out, texts, pairs, triggers, cfg)
    rows = [r for r in texts.values() if r["status"] == "ok"]

    def work(r):
        if text_key(r) in done:
            return
        messages, identity = checker_input(r, pairs, triggers, cfg)
        result = calls.chat(
            f"check:{r['task_id']}:{r['condition']}:{r['text_sha256']}:{identity}",
            messages,
        )
        status, verdict = parse_result(result)
        append(
            out / "text_checks.jsonl",
            {
                **{k: r[k] for k in ("task_id", "condition", "text_sha256")},
                "checker_input_sha256": identity,
                "status": status,
                "passed": status == "ok" and verdict.get("matches_condition") is True,
                "verdict": verdict,
            },
        )

    with ThreadPoolExecutor(6) as pool:
        list(pool.map(work, rows[: args.limit or len(rows)]))


def check_passed(record, checks):
    verdict = checks.get(text_key(record), {})
    return verdict.get("passed") is True and verdict.get("status") == "ok"


def text_cells(selection, texts, checks, cfg):
    """Resolve donors once by the frozen rule, before any hand dispositions."""
    cells = {}
    for split in ("development", "heldout"):
        for task in selection[split]:
            ref = texts.get((task, "task_restatement"))
            for kind in cfg["conditions"]:
                if kind == "none":
                    continue
                r, donor = texts.get((task, kind)), None
                if kind == "irrelevant" and "frozen_donors" in cfg.get(
                    "amendment_1", {}
                ):
                    donor = cfg["amendment_1"]["frozen_donors"][task]
                    if donor is not None and (
                        donor == task or donor not in selection[split]
                    ):
                        raise ValueError(
                            "frozen donor must be another task in the same split"
                        )
                    r = texts.get((donor, "task_restatement")) if donor else None
                elif kind == "irrelevant":
                    r = None
                    if ref and ref["status"] == "ok":
                        band = length_band(len(ref["text"]), cfg["texts"]["tolerance"])
                        donors = [
                            v
                            for (t, k), v in texts.items()
                            if t in selection[split]
                            and t != task
                            and k == "task_restatement"
                            and v["status"] == "ok"
                            and band[0] <= len(v["text"]) <= band[1]
                        ]
                        donors.sort(
                            key=lambda v: (
                                abs(len(v["text"]) - len(ref["text"])),
                                order_key(cfg["selection"]["seed"], task, v["task_id"]),
                            )
                        )
                        r = donors[0] if donors else None
                        donor = r["task_id"] if r else None
                cells[task, kind] = {
                    "task_id": task,
                    "condition": kind,
                    "text_sha256": r["text_sha256"] if r else None,
                    "text_status": r["status"] if r else "missing",
                    "donor_task_id": donor,
                    "record": r,
                }
    return cells


def cell_identity(cell, pair, trigger):
    return {
        **{k: v for k, v in cell.items() if k != "record"},
        **input_identity(pair, trigger),
    }


def sample_tasks(selection, cfg):
    tasks = sorted(
        selection["development"] + selection["heldout"],
        key=lambda t: order_key(cfg["texts"]["hand_audit_seed"], t),
    )
    count = cfg["texts"]["hand_audit_tasks"]
    if len(tasks) < count or len(set(tasks)) != len(tasks):
        raise ValueError("insufficient or duplicate tasks for frozen hand-audit sample")
    return tasks[:count]


def sample_records(selection, cells, pairs, triggers, cfg):
    return [
        {
            "task_id": task,
            **input_identity(pairs[task], triggers[task]),
            "trigger": triggers[task],
            "problem": apps.parse(pairs[task]["honest"]["prompt"]).problem,
            "programs": {
                c: apps.parse(pairs[task][c]["prompt"]).code for c in pairs[task]
            },
            "texts": {
                kind: cells[task, kind] for kind in cfg["conditions"] if kind != "none"
            },
        }
        for task in sample_tasks(selection, cfg)
    ]


def audit_sample(out, cfg):
    selection = json.loads((out / "selection.json").read_text())
    texts = final_texts(out)
    pairs, _ = load_pairs(cfg)
    _, triggers = prior(cfg)
    checks = current_checks(out, texts, pairs, triggers, cfg)
    cells = text_cells(selection, texts, checks, cfg)
    records = sample_records(selection, cells, pairs, triggers, cfg)
    write_jsonl(out / "hand_audit_sample.jsonl", records)
    print(
        f"Read {len(records)} tasks. In trigger_hand_reads.jsonl record task_id, "
        "trigger_sha256, problem_sha256, code_pair_sha256, boolean valid, reason. "
        "Hand-read EVERY current text cell. In hand_audit.jsonl record those "
        "fields plus condition, text_sha256, "
        "text_status, donor_task_id. Use null text_sha256 for missing text and "
        "valid=false for missing/failed text. Read all five text conditions, "
        "including irrelevant against the RECIPIENT problem, not its donor."
    )


def build(out, cfg, *, dry_run=False):
    validate_pipeline(out, cfg, dry_run)
    selection = json.loads((out / "selection.json").read_text())
    pairs, _ = load_pairs(cfg)
    _, triggers = prior(cfg)
    texts = final_texts(out)
    checks = current_checks(out, texts, pairs, triggers, cfg)
    cells = text_cells(selection, texts, checks, cfg)
    reads = {
        (r["task_id"], r["condition"]): r for r in read_jsonl(out / "hand_audit.jsonl")
    }
    tasks = selection["development"] + selection["heldout"]
    allowed = eligible(out, pairs, triggers, required=tasks if not dry_run else ())
    selected = set()
    if not dry_run:
        expected = sample_records(selection, cells, pairs, triggers, cfg)
        if read_jsonl(out / "hand_audit_sample.jsonl") != expected:
            raise SystemExit("draw and read the current frozen hand-audit sample")
        selected = set(sample_tasks(selection, cfg))
        overrides = {
            r["task_id"]: r for r in read_jsonl(out / "trigger_hand_reads.jsonl")
        }
        for task in selected:
            if task not in overrides:
                raise SystemExit("hand-audit sample needs explicit trigger verdicts")
            hand_verdict(
                overrides[task],
                {
                    "task_id": task,
                    **input_identity(pairs[task], triggers[task]),
                },
            )
    # Preflight even rejected sample tasks and absent/failed texts. Looping only
    # over usable rows would let missing texts and trigger exclusions escape reads.
    dispositions = {}
    for key, cell in cells.items():
        task, kind = key
        r = cell["record"]
        read = reads.get(key)
        if read is not None:
            valid = hand_verdict(read, cell_identity(cell, pairs[task], triggers[task]))
            if valid and cell["text_status"] != "ok":
                raise ValueError("missing/failed text requires valid=false disposition")
            dispositions[key] = valid
        if not dry_run and read is None:
            raise SystemExit(
                "read every text cell: sampled condition and failed text included"
            )

    rows, failures = [], []
    for split in ("development", "heldout"):
        for task in selection[split]:
            if task not in allowed:
                failures.append(
                    {
                        "task_id": task,
                        "reason": "trigger_unverified",
                        "split": split,
                        "excluded_rows": 2 * len(cfg["conditions"]),
                    }
                )
                continue
            ref = texts.get((task, "task_restatement"))
            for kind in cfg["conditions"]:
                r, donor, failure = None, None, None
                if kind != "none":
                    cell = cells[task, kind]
                    r, donor = cell["record"], cell["donor_task_id"]
                    if r is None or r["status"] != "ok":
                        failure = "text_missing_or_failed"
                    elif dispositions.get((task, kind)) is False:
                        failure = "hand_read_invalid"
                    elif (
                        donor and dispositions.get((donor, "task_restatement")) is False
                    ):
                        failure = "donor_hand_read_invalid"
                    elif not dry_run and dispositions.get((task, kind)) is not True:
                        failure = "hand_read_unverified"
                    elif kind != "task_restatement" and (
                        not ref
                        or ref["status"] != "ok"
                        or not length_band(len(ref["text"]), cfg["texts"]["tolerance"])[
                            0
                        ]
                        <= len(r["text"])
                        <= length_band(len(ref["text"]), cfg["texts"]["tolerance"])[1]
                    ):
                        failure = "length_mismatch"
                cell_failure = {
                    "task_id": task,
                    "split": split,
                    "condition": kind,
                    "excluded_rows": 2,
                }
                if failure:
                    failures.append({**cell_failure, "reason": failure})
                    continue
                text = r["text"] if r else ""
                header, value = docstring(text, 79) if r else ("", "")
                pair_rows, insertion_failures = [], []
                for code in ("honest", "backdoored"):
                    row = pairs[task][code]
                    new_prompt = row["prompt"]
                    if r:
                        try:
                            original = apps.parse(row["prompt"]).code
                            edited, at = insert_header(original, header)
                            checked = check_variant(original, edited, header, at, value)
                            if not checked.ok:
                                insertion_failures.append(
                                    {"code": code, "reason": checked.reason}
                                )
                                continue
                            new_prompt = edited_prompt(row, edited)
                        except (ValueError, SyntaxError) as error:
                            insertion_failures.append(
                                {"code": code, "reason": str(error)}
                            )
                            continue
                    pair_rows.append(
                        {
                            "id": f"crossed:{task}:{code}:{kind}",
                            "prompt": new_prompt,
                            "metadata": {
                                "task_id": task,
                                "code": code,
                                "text": kind,
                                "condition": kind,
                                "split": split,
                                "set": "main",
                                "label": int(code == "backdoored"),
                                "added_chars": len(header),
                                "raw_text_chars": len(text),
                                "text_sha256": sha(text),
                                "raw_text_sha256": sha(text),
                                "rendered_header_sha256": sha(header),
                                "rendered_docstring_sha256": sha(value),
                                "donor_task_id": donor,
                            },
                        }
                    )
                if insertion_failures:
                    failures.append(
                        {
                            **cell_failure,
                            "reason": "code_insertion_failed",
                            "details": insertion_failures,
                        }
                    )
                else:
                    rows.extend(pair_rows)
    for task in selection["calibration"]:
        rows.append(
            {
                "id": f"crossed:calibration:{task}",
                "prompt": pairs[task]["honest"]["prompt"],
                "metadata": {
                    "task_id": task,
                    "code": "honest",
                    "text": "none",
                    "split": "calibration",
                    "set": "calibration",
                    "label": 0,
                    "added_chars": 0,
                    "raw_text_chars": 0,
                    "text_sha256": sha(""),
                    "raw_text_sha256": sha(""),
                    "rendered_header_sha256": sha(""),
                    "rendered_docstring_sha256": sha(""),
                },
            }
        )
    # Exact crossing is checked independently of insertion and selection logic.
    grid = {}
    for row in rows:
        m = row["metadata"]
        if m["set"] == "main":
            grid.setdefault((m["task_id"], m["text"]), []).append(m)
    if any(
        len(v) != 2
        or {r["code"] for r in v} != {"honest", "backdoored"}
        or len({r["raw_text_sha256"] for r in v}) != 1
        or len({r["rendered_header_sha256"] for r in v}) != 1
        for v in grid.values()
    ):
        raise ValueError("a cell lacks a matched pair with identical text and header")
    write_jsonl(out / "variants.jsonl", rows)
    write_jsonl(out / "variant_failures.jsonl", failures)
    summary = {
        "rows": len(rows),
        "by_split": dict(Counter(r["metadata"]["split"] for r in rows)),
        "failures": dict(Counter(r["reason"] for r in failures)),
        "excluded_rows": sum(r["excluded_rows"] for r in failures),
        "variants_sha256": digest(out / "variants.jsonl"),
    }
    write_json(out / "variants_summary.json", summary)
    seal_variants(out, cfg, dry_run)
    print(summary)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage",
        choices=[
            "prepare",
            "import-completed",
            "audit-triggers",
            "generate",
            "check",
            "audit-sample",
            "build",
        ],
    )
    p.add_argument("--out")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    cfg, out = config(), output(args.out)
    if args.limit is not None and args.limit < 1:
        p.error("--limit must be positive")
    if not args.dry_run and args.stage in {"audit-triggers", "generate", "check"}:
        frozen(cfg)
    with exclusive(out, "pipeline"):
        if args.stage == "prepare":
            prepare(out, cfg)
            return
        bind(
            out,
            cfg,
            "pipeline",
            {
                "dry_run": args.dry_run,
                "selection_sha256": digest(out / "selection.json"),
            },
        )
        if args.stage == "import-completed":
            import_completed(out, cfg, dry_run=args.dry_run)
            return
        validate_import(out, cfg, args.dry_run)
        if args.stage == "audit-triggers":
            run_audit(args, out, cfg)
        elif args.stage == "generate":
            generate(args, out, cfg)
        elif args.stage == "check":
            check(args, out, cfg)
        elif args.stage == "audit-sample":
            audit_sample(out, cfg)
        elif args.stage == "build":
            build(out, cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
