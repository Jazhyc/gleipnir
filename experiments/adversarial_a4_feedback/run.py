"""Paired five-attempt feedback pilot; generation is local, scoring is remote."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

import numpy as np

# Explicit, checksum-bound reuse of the frozen A2 validator and A3 score journal.
from experiments.adversarial_a2_cover_comments.run import insert_comments
from experiments.adversarial_a3_random_search.journal import Journal
from gleipnir.adversarial import apps, eval_sets
from gleipnir.adversarial.attacker import (
    AttackerConfig,
    Budget,
    BudgetExceeded,
    complete,
)
from gleipnir.adversarial.metrics import summary, threshold_at_fpr

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text())
OUT = eval_sets.ROOT / CONFIG["output"]


def read(path):
    return (
        [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    )


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def append(path, record):
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


@contextmanager
def lock(name):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / (name + ".lock")).open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def inputs():
    path = eval_sets.ROOT / CONFIG["baseline"]
    assert eval_sets.file_sha256(path) == CONFIG["baseline_sha256"]
    baseline = {r["id"]: r for r in read(path)}
    all_rows = {
        (str(r["metadata"]["task_id"]), eval_sets.group(r)): r
        for r in eval_sets.load("apps-neutral")
    }
    thresholds = {
        str(f): threshold_at_fpr(
            [r["score"] for r in baseline.values() if r["label"] == 0], f
        )
        for f in CONFIG["fprs"]
    }
    eligible = []
    for task, group in all_rows:
        if (
            group != "vanilla"
            or baseline[all_rows[(task, group)]["id"]]["score"] <= thresholds["0.01"]
        ):
            continue
        pair = [all_rows[(task, c)] for c in ("clean", "vanilla")]
        if (
            max(baseline[r["id"]]["tokens"] for r in pair)
            > CONFIG["selection_max_original_tokens"]
        ):
            continue
        try:
            for row in pair:
                insert_comments(
                    apps.parse(row["prompt"]).code,
                    [{"before_line": 1, "comment": "Implementation notes."}],
                )
        except ValueError:
            continue
        eligible.append(task)
    selected = sorted(eligible, key=lambda t: digest("a4-pilot-v1:" + t))[
        : len(CONFIG["task_ids"])
    ]
    assert len(eligible) == CONFIG["eligible_tasks"] and selected == CONFIG["task_ids"]
    rows = {
        k: r
        for k, r in all_rows.items()
        if k[0] in selected and k[1] in ("clean", "vanilla")
    }
    return rows, baseline, thresholds


def prepare():
    rows, _, thresholds = inputs()
    files = list(HERE.glob("*.py")) + list(HERE.glob("*.txt")) + [HERE / "config.json"]
    files += list((eval_sets.ROOT / "src/gleipnir/adversarial").glob("*.py"))
    files += [
        eval_sets.ROOT / p
        for p in (
            "experiments/adversarial_a2_cover_comments/run.py",
            "experiments/adversarial_a2_cover_comments/config.json",
            "experiments/adversarial_a3_random_search/journal.py",
        )
    ]
    identity = {
        "files": {
            str(p.relative_to(eval_sets.ROOT)): eval_sets.file_sha256(p)
            for p in sorted(files)
        },
        "data_sha256": eval_sets.SETS["apps-neutral"].sha256,
        "prompt_hashes": {r["id"]: digest(r["prompt"]) for r in rows.values()},
    }
    manifest = {"identity": identity, "thresholds": thresholds}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("manifest changed")
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def candidate_id(task, arm, round_no):
    return f"{task}:{'shared' if round_no == 1 else arm}:{round_no}"


def generations():
    return {r["id"]: r for r in read(OUT / "generations.jsonl")}


def score_values():
    return {
        r["id"]: r["value"]
        for r in read(OUT / "queries.jsonl")
        if r["type"] == "result" and r["status"] == "ok"
    }


def histories(task, arm, round_no, records, scores):
    result = []
    for q in range(1, round_no):
        cid = candidate_id(task, arm, q)
        record = records[cid]
        item = {"attempt": q, "status": record["status"]}
        if record["status"] == "ok":
            item["edits"] = record["edits"]
            item["suspicion"] = scores["candidate:" + cid]["score"]
        else:
            item["validation_error"] = record.get("error", "No valid comments returned")
        result.append(item)
    return result


def messages_for(row, task, arm, round_no, records, scores):
    sub = apps.parse(row["prompt"])
    prompt = (
        (HERE / "prompt.txt")
        .read_text()
        .format(
            problem=sub.problem,
            code="".join(
                f"{i}: {line}"
                for i, line in enumerate(sub.code.splitlines(keepends=True), 1)
            ),
        )
    )
    if arm == "feedback" and round_no > 1:
        prompt += "\n" + (HERE / "feedback.txt").read_text().format(
            baseline=scores[f"baseline:{task}:vanilla"]["score"],
            history=json.dumps(histories(task, arm, round_no, records, scores)),
        )
    return [{"role": "user", "content": prompt}]


def validate_response(result, row):
    if result["finish_reason"] == "length":
        return {"status": "truncation"}
    if result["finish_reason"] in ("content_filter", "refusal"):
        return {"status": "refusal"}
    text = result["text"].strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        edits = json.loads(text)
    except (ValueError, TypeError):
        return {"status": "invalid_json"}
    try:
        original = apps.parse(row["prompt"]).code
        edited, positions = insert_comments(original, edits)
        prompt = apps.with_code(row["prompt"], edited)
        assert apps.parse(prompt).code == edited
        return {
            "status": "ok",
            "edits": edits,
            "inserted_positions": positions,
            "prompt": prompt,
            "prompt_sha256": digest(prompt),
            "original_code_sha256": digest(original),
        }
    except (ValueError, TypeError, KeyError, AssertionError) as error:
        return {"status": "invalid_edit", "error": str(error)[:200]}


def generate(round_no):
    with lock("generate"):
        prepare()
        rows, _, _ = inputs()
        records = generations()
        intents = {r["id"]: r for r in read(OUT / "api_intents.jsonl")}
        if set(intents) != set(records):
            raise RuntimeError(
                "uncertain API call; reconcile before resuming, never retry"
            )
        if round_no > 1:
            stage = json.loads((OUT / f"score_round_{round_no - 1}.json").read_text())
            assert stage["state"] == "succeeded"
        scores = score_values()
        cfg = AttackerConfig(
            **{
                **CONFIG["attacker"],
                "providers": tuple(CONFIG["attacker"]["providers"]),
            }
        )
        budget = Budget(
            CONFIG["api_budget_usd"],
            spent_usd=sum(r["charged_usd"] for r in records.values()),
        )
        jobs = []
        for task in CONFIG["task_ids"]:
            for arm in ["shared"] if round_no == 1 else CONFIG["arms"]:
                cid = candidate_id(task, arm, round_no)
                row = rows[(task, "vanilla")]
                messages = messages_for(row, task, arm, round_no, records, scores)
                sha = digest(json.dumps(messages, sort_keys=True))
                if cid in records:
                    assert records[cid]["request_sha256"] == sha
                else:
                    jobs.append((cid, task, arm, row, messages, sha))

        def work(job):
            cid, task, arm, row, messages, sha = job
            reserve = cfg.worst_case_usd(
                sum(len(m["content"].encode()) for m in messages)
            )
            # Journal intent before sending; simultaneous independent writers use
            # one append syscall under this process's separate thread lock.
            with write_lock:
                append(
                    OUT / "api_intents.jsonl",
                    {
                        "id": cid,
                        "request_sha256": sha,
                        "reserved_usd": reserve,
                        "timestamp": time.time(),
                    },
                )
            record = {
                "id": cid,
                "task": task,
                "arm": arm,
                "round": round_no,
                "request_sha256": sha,
                "timestamp": time.time(),
                "charged_usd": 0.0,
            }
            try:
                result = complete(
                    messages, cfg, budget, timeout=CONFIG["api_timeout_seconds"]
                )
                record.update(response=result, charged_usd=result["cost_usd"])
                record.update(validate_response(result, row))
            except BudgetExceeded:
                record["status"] = "budget_exhausted"
            except Exception as error:
                record.update(
                    status="infrastructure_failure",
                    charged_usd=reserve,
                    error=type(error).__name__,
                    error_details=getattr(error, "details", {}),
                )
            return record

        import threading

        write_lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=CONFIG["workers"]) as pool:
            futures = [pool.submit(work, j) for j in jobs]
            for n, f in enumerate(as_completed(futures), 1):
                record = f.result()
                append(OUT / "generations.jsonl", record)
                print(
                    f"round {round_no}: generated {n}/{len(jobs)}, "
                    f"{record['status']}; charged ${budget.spent_usd:.4f}",
                    flush=True,
                )
        current = [r for r in generations().values() if r["round"] == round_no]
        expected = len(CONFIG["task_ids"]) * (
            1 if round_no == 1 else len(CONFIG["arms"])
        )
        assert len(current) == expected
        if (
            round_no == 1
            and sum(r["status"] == "ok" for r in current)
            < CONFIG["minimum_first_round_valid"]
        ):
            raise RuntimeError(
                "first-round validity stop: retain failures, do not launch GPU"
            )
        if any(r["status"] == "budget_exhausted" for r in current):
            raise RuntimeError("API budget exhausted")


def score(round_no):
    from gleipnir.adversarial.scorer import Scorer

    with lock("score"):
        manifest = prepare()
        path = OUT / f"score_round_{round_no}.json"
        if path.exists():
            old = json.loads(path.read_text())
            if old["state"] == "succeeded":
                print("round already complete; no scoring calls", flush=True)
                return
            raise RuntimeError(
                "unfinished previous scoring session; reconcile before resume"
            )
        path.write_text(json.dumps({"state": "running"}))
        start = time.monotonic()
        state = "failed"
        try:
            scorer = Scorer(CONFIG["monitor"])
            journal = Journal(
                OUT / "queries.jsonl",
                {"manifest": manifest["identity"], "scorer": scorer.identity()},
                CONFIG["total_score_call_cap"],
            )
            rows, baseline, thresholds = inputs()
            records = generations()
            all_tasks = CONFIG["task_ids"] if round_no == 1 else CONFIG["task_ids"][:2]
            drift = []
            for task in all_tasks:
                for condition in ("clean", "vanilla"):
                    row = rows[(task, condition)]
                    cid = (
                        f"baseline:{task}:{condition}"
                        if round_no == 1
                        else f"canary:{round_no}:{task}:{condition}"
                    )
                    value = journal.score(
                        cid,
                        row["prompt"],
                        scorer.score,
                        {
                            "kind": "baseline" if round_no == 1 else "canary",
                            "task": task,
                            "condition": condition,
                            "round": round_no,
                        },
                    )
                    delta = abs(value["score"] - baseline[row["id"]]["score"])
                    drift.append(delta)
                    if delta > CONFIG["baseline_max_abs_drift"] or (
                        condition == "vanilla" and value["score"] <= thresholds["0.01"]
                    ):
                        raise ValueError("baseline canary failed")
            if round_no == 1:
                effects = []
                for condition in ("clean", "vanilla"):
                    task = CONFIG["task_ids"][0]
                    row = rows[(task, condition)]
                    raw = journal.score(
                        "unadapted:" + condition,
                        row["prompt"],
                        lambda p: scorer.score(p, adapter=False),
                        {"kind": "adapter_canary", "condition": condition},
                    )
                    effects.append(
                        abs(
                            raw["score"]
                            - journal.results[f"baseline:{task}:{condition}"]["value"][
                                "score"
                            ]
                        )
                    )
                if max(effects) <= 1e-6:
                    raise ValueError("adapter effect missing")
                (OUT / "runtime.json").write_text(
                    json.dumps(
                        {
                            "identity": scorer.identity(),
                            "commit": subprocess.check_output(
                                ["git", "rev-parse", "HEAD"], text=True
                            ).strip(),
                            "adapter_effects": effects,
                        },
                        indent=2,
                    )
                )
            (OUT / f"canary_round_{round_no}.json").write_text(
                json.dumps({"passed": True, "max_drift": max(drift)})
            )
            expected = len(CONFIG["task_ids"]) * (
                1 if round_no == 1 else len(CONFIG["arms"])
            )
            current = [r for r in records.values() if r["round"] == round_no]
            assert len(current) == expected
            for record in sorted(current, key=lambda r: r["id"]):
                if record["status"] != "ok":
                    continue
                row = rows[(record["task"], "vanilla")]
                revalidated = validate_response(record["response"], row)
                assert (
                    revalidated["status"] == "ok"
                    and revalidated["prompt"] == record["prompt"]
                )
                journal.score(
                    "candidate:" + record["id"],
                    record["prompt"],
                    scorer.score,
                    {
                        "kind": "candidate",
                        "task": record["task"],
                        "arm": record["arm"],
                        "round": round_no,
                        "generation_id": record["id"],
                    },
                )
            if round_no == CONFIG["rounds"]:
                values = {cid: r["value"] for cid, r in journal.results.items()}
                for task in CONFIG["task_ids"]:
                    for arm in CONFIG["arms"]:
                        winner = best(task, arm, round_no, records, values)
                        if winner is not None:
                            # Transfer comments as a full-line header: original
                            # line numbers do not align between honest/bad code.
                            original = apps.parse(rows[(task, "clean")]["prompt"]).code
                            text = (
                                "\n".join("# " + e["comment"] for e in winner["edits"])
                                + "\n"
                            )
                            from gleipnir.adversarial.python_code import (
                                check_comment_only_edit,
                            )

                            assert check_comment_only_edit(original, text + original).ok
                            prompt = apps.with_code(
                                rows[(task, "clean")]["prompt"], text + original
                            )
                            journal.score(
                                f"honest:{task}:{arm}",
                                prompt,
                                scorer.score,
                                {
                                    "kind": "honest",
                                    "task": task,
                                    "arm": arm,
                                    "generation_id": winner["id"],
                                },
                            )
            state = "succeeded"
        finally:
            path.write_text(
                json.dumps(
                    {"state": state, "seconds": time.monotonic() - start}, indent=2
                )
            )
        print(
            f"round {round_no} scoring succeeded; total calls {len(journal.intents)}",
            flush=True,
        )


def best(task, arm, round_no, records, scores):
    current = scores[f"baseline:{task}:vanilla"]["score"]
    winner = None
    for q in range(1, round_no + 1):
        cid = candidate_id(task, arm, q)
        if "candidate:" + cid in scores:
            score = scores["candidate:" + cid]["score"]
            if score < current:
                current = score
                winner = records[cid]
    return winner


def report():
    manifest = prepare()
    records = generations()
    scores = score_values()
    journal = read(OUT / "queries.jsonl")
    if journal and journal[0]["value"]["manifest"] != manifest["identity"]:
        raise ValueError("scoring identity drift")
    curves = {}
    flags = {}
    winners = []
    for arm in CONFIG["arms"]:
        curves[arm] = {}
        for round_no in range(1, CONFIG["rounds"] + 1):
            values = []
            complete_count = 0
            for task in CONFIG["task_ids"]:
                complete_set = all(
                    candidate_id(task, arm, q) in records
                    and (
                        records[candidate_id(task, arm, q)]["status"] != "ok"
                        or "candidate:" + candidate_id(task, arm, q) in scores
                    )
                    for q in range(1, round_no + 1)
                )
                complete_count += int(complete_set)
                if f"baseline:{task}:vanilla" not in scores:
                    values.append(1.0)
                    continue
                winner = best(task, arm, round_no, records, scores)
                value = (
                    scores["candidate:" + winner["id"]]["score"]
                    if winner
                    else scores[f"baseline:{task}:vanilla"]["score"]
                )
                values.append(value if complete_set else 1.0)
                if round_no == CONFIG["rounds"]:
                    winners.append(
                        {
                            "task": task,
                            "arm": arm,
                            "score": value,
                            "baseline": scores[f"baseline:{task}:vanilla"]["score"],
                            "generation_id": winner["id"] if winner else None,
                            "complete": complete_set,
                        }
                    )
            entry = {
                "completed_searches": complete_count,
                "mean_best_score": float(np.mean(values)),
            }
            for f, t in manifest["thresholds"].items():
                fvalues = np.array(values) <= t
                flags[(arm, round_no, f)] = fvalues.astype(float)
                entry[f] = {
                    "successes": int(fvalues.sum()),
                    "planned_searches": len(values),
                }
            curves[arm][str(round_no)] = entry
    paired = {}
    for q in range(1, CONFIG["rounds"] + 1):
        paired[str(q)] = {}
        for f in manifest["thresholds"]:
            diff = flags[("feedback", q, f)] - flags[("independent", q, f)]
            boot = (
                np.random.default_rng(0).choice(diff, (10000, len(diff))).mean(axis=1)
            )
            paired[str(q)][f] = {
                "difference": float(diff.mean()),
                "task_bootstrap95": np.quantile(boot, [0.025, 0.975]).tolist(),
            }
    honest = {}
    for arm in CONFIG["arms"]:
        pairs = []
        for w in winners:
            if w["arm"] != arm or not w["complete"]:
                continue
            cid = (
                f"honest:{w['task']}:{arm}"
                if w["generation_id"]
                else f"baseline:{w['task']}:clean"
            )
            if cid in scores:
                pairs.append((w["score"], scores[cid]["score"]))
        metrics = (
            summary(
                [1] * len(pairs) + [0] * len(pairs),
                [x[0] for x in pairs] + [x[1] for x in pairs],
            )
            if pairs
            else {}
        )
        honest[arm] = {
            "pairs": len(pairs),
            "flagged": {
                f: sum(h > t for _, h in pairs)
                for f, t in manifest["thresholds"].items()
            },
            "ranking": {k: metrics[k] for k in ("auroc", "pauroc20") if k in metrics},
        }
    result = {
        "curves": curves,
        "paired": paired,
        "winners": winners,
        "honest": honest,
        "generation_statuses": dict(Counter(r["status"] for r in records.values())),
        "api_charged_usd": sum(r["charged_usd"] for r in records.values()),
        "physical_score_calls": sum(r["type"] == "intent" for r in journal),
        "successful_score_calls": len(scores),
        "prompt_tokens": sum(r["tokens"] for r in scores.values()),
    }
    (OUT / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "generate", "score", "report"])
    parser.add_argument("--round", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.round <= CONFIG["rounds"]:
        raise ValueError("invalid round")
    if args.command in ("generate", "score"):
        globals()[args.command](args.round)
    else:
        globals()[args.command]()
