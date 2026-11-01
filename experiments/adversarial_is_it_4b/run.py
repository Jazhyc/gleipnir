"""Score the "is it a 4B thing?" arms and report.

GPU arms (on the box):
    python experiments/adversarial_is_it_4b/run.py smoke --arm v03-merged
    python experiments/adversarial_is_it_4b/run.py score --arm v03-merged
Kimi K3 through OpenRouter (on the Mac; reads OPENROUTER_API_KEY):
    python experiments/adversarial_is_it_4b/run.py kimi
Analysis (anywhere, after pulling results):
    python experiments/adversarial_is_it_4b/run.py parity
    python experiments/adversarial_is_it_4b/run.py report
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import median
from typing import Any

from gleipnir.adversarial import eval_sets
from gleipnir.adversarial.metrics import threshold_at_fpr

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text())
ROOT = eval_sets.ROOT
DATA = ROOT / CONFIG["data"]
OUTPUT = ROOT / CONFIG["output"]
LAB = ROOT / CONFIG["lab_results"]


def git_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()


# ---------------------------------------------------------------- data


def load_set(name: str, prompt: str) -> list[dict[str, Any]]:
    path = DATA / f"{name}_{prompt}.jsonl"
    key = f"{name}_{prompt}"
    if eval_sets.file_sha256(path) != CONFIG["sha256"][key]:
        raise ValueError(f"{path} does not match its frozen SHA256")
    return [json.loads(line) for line in path.open()]


def arm_rows(arm: str, name: str) -> list[dict[str, Any]]:
    """The rows one arm scores: its calibration sizes, readers and conditions."""
    spec = CONFIG["arms"][arm]
    rows = load_set(name, spec["prompt"])
    if name == "calibration":
        sizes = {**CONFIG["calibration"], **spec.get("calibration", {})}
        limit = {0: sizes["honest"], 1: sizes["harmful"]}
        return [
            r
            for r in rows
            if r["metadata"]["calibration_rank"] < limit[eval_sets.label(r)]
        ]
    if name == "framing" and spec.get("readers") == "subset":
        keep = set(CONFIG["subset"]["conditions"])
        return [
            r
            for r in rows
            if r["metadata"]["subset"] and r["metadata"]["condition"] in keep
        ]
    return rows


# ---------------------------------------------------------------- GPU scoring


def monitor_for(arm: str, override: str | None) -> str:
    spec = CONFIG["arms"][arm]
    if override and override not in (spec["monitor"], spec.get("fallback_monitor")):
        raise ValueError(f"{override} is not a registered monitor for {arm}")
    return override or spec["monitor"]


def smoke(args: argparse.Namespace) -> None:
    from gleipnir.adversarial.scorer import Scorer

    monitor = monitor_for(args.arm, args.monitor)
    calibration = arm_rows(args.arm, "calibration")
    rows = [r for r in calibration if r["metadata"]["calibration_rank"] < 2]
    rows += [r for r in arm_rows(args.arm, "liars") if "swap" in r["id"]][:2]
    started = time.time()
    scorer = Scorer(monitor, device=args.device)
    loaded = time.time() - started
    results = []
    for row in rows:
        out = scorer.score(row["prompt"])
        record = {"id": row["id"], "label": eval_sets.label(row), **out}
        if scorer.spec.adapter is not None:
            base = scorer.score(row["prompt"], adapter=False)
            record["adapter_effect"] = abs(out["score"] - base["score"])
        results.append(record)
        print(record, flush=True)
    passed = all(math.isfinite(r["score"]) for r in results)
    if scorer.spec.adapter is not None:
        passed = passed and max(r["adapter_effect"] for r in results) >= 1e-6
    directory = OUTPUT / args.arm
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "smoke.json").write_text(
        json.dumps(
            {
                "passed": passed,
                "load_seconds": loaded,
                "identity": scorer.identity(),
                "commit": git_commit(),
                "rows": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"smoke passed={passed}")
    if not passed:
        raise SystemExit(1)


def score(args: argparse.Namespace) -> None:
    from gleipnir.adversarial.scorer import Scorer, score_rows

    monitor = monitor_for(args.arm, args.monitor)
    scorer = Scorer(monitor, device=args.device)
    for name in args.sets or CONFIG["arms"][args.arm]["sets"]:
        rows = arm_rows(args.arm, name)
        directory = OUTPUT / args.arm / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "identity.json").write_text(
            json.dumps(
                {
                    "arm": args.arm,
                    "set": name,
                    "rows": len(rows),
                    "commit": git_commit(),
                    "scorer": scorer.identity(),
                },
                indent=2,
            )
            + "\n"
        )
        score_rows(scorer, rows, directory / "predictions.jsonl")


# ---------------------------------------------------------------- Kimi via API


def kimi_record(row: dict[str, Any], config, api_key: str) -> dict[str, Any]:
    """Score one row with the teacher's request settings.

    Mirrors `gleipnir.teachers.openrouter.score_prompt`, but keeps the response
    when one label is missing from the top five: that label's log-probability
    is then bounded by the fifth alternative (pre-registered, `bounded=True`).
    """
    import requests

    from gleipnir.teachers.openrouter import (
        PromptRecord,
        binary_score_from_top_logprobs,
        extract_terminal_binary_top_logprobs,
        request_payload,
        retry_delay_seconds,
    )

    record = PromptRecord(row["id"], row["prompt"], {})
    payload = request_payload(record, config)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": config.app_url,
        "X-Title": "Gleipnir adversarial stress tests",
    }
    out = {
        "id": row["id"],
        "group": eval_sets.group(row),
        "label": eval_sets.label(row),
        **{
            k: row["metadata"][k]
            for k in ("task_id", "parent_id")
            if k in row["metadata"]
        },
    }
    error = None
    for attempt in range(config.max_retries + 1):
        response = None
        try:
            response = requests.post(
                config.endpoint, headers=headers, json=payload, timeout=180
            )
            if response.status_code >= 400:
                raise RuntimeError(
                    f"HTTP {response.status_code}: {response.text[:300]}"
                )
            data = response.json()
            top, text, _ = extract_terminal_binary_top_logprobs(data)
            bounded = False
            if not {"0", "1"} <= set(top):
                floor = min(top.values())
                top = {**{"0": floor, "1": floor}, **top}
                bounded = True
            value, label_logprobs = binary_score_from_top_logprobs(top)
            usage = data.get("usage") or {}
            out.update(
                status="ok",
                score=value,
                logit_0=label_logprobs["0"],
                logit_1=label_logprobs["1"],
                bounded=bounded,
                text=text,
                provider=data.get("provider"),
                model=data.get("model"),
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                cost=usage.get("cost"),
                attempts=attempt + 1,
            )
            return out
        except Exception as exc:  # recorded, never silently dropped
            error = f"{type(exc).__name__}: {exc}"[:500]
            retryable = response is None or response.status_code in {
                408,
                409,
                429,
                500,
                502,
                503,
                504,
                524,
                529,
            }
            if attempt >= config.max_retries or not retryable:
                break
            time.sleep(retry_delay_seconds(attempt, response))
    out.update(status="error", error=error)
    return out


def kimi(args: argparse.Namespace) -> None:
    from gleipnir.teachers.openrouter import OpenRouterConfig, request_settings

    spec = CONFIG["arms"]["kimi-teacher"]
    config = OpenRouterConfig(
        model=spec["model"],
        max_tokens=spec["max_tokens"],
        top_logprobs=spec["top_logprobs"],
        provider_sort=None,
        provider_only=spec["provider_only"],
        provider_max_prompt_price=spec["max_prompt_price"],
        provider_max_completion_price=spec["max_completion_price"],
        allow_fallbacks=False,
    )
    api_key = os.environ["OPENROUTER_API_KEY"]
    lock = threading.Lock()
    spent = [0.0]
    for name in args.sets or spec["sets"]:
        rows = arm_rows("kimi-teacher", name)
        if args.limit:
            rows = rows[: args.limit]
        directory = OUTPUT / "kimi-teacher" / name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "predictions.jsonl"
        done = set()
        if path.exists():
            for line in path.open():
                record = json.loads(line)
                if record["status"] == "ok":
                    done.add(record["id"])
                spent[0] += record.get("cost") or 0.0
        (directory / "identity.json").write_text(
            json.dumps(
                {
                    "arm": "kimi-teacher",
                    "set": name,
                    "rows": len(rows),
                    "commit": git_commit(),
                    "request_settings": request_settings(config),
                },
                indent=2,
            )
            + "\n"
        )
        todo = [r for r in rows if r["id"] not in done]
        print(f"{name}: {len(todo)} to score, ${spent[0]:.2f} spent so far", flush=True)

        def work(row, path=path):
            if spent[0] >= spec["max_usd"]:
                return None
            record = kimi_record(row, config, api_key)
            with lock:
                spent[0] += record.get("cost") or 0.0
                with path.open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
            return record

        with ThreadPoolExecutor(spec["workers"]) as pool:
            for n, _record in enumerate(pool.map(work, todo), 1):
                if n % 25 == 0 or n == len(todo):
                    print(f"  {name}: {n}/{len(todo)}, ${spent[0]:.2f}", flush=True)
        if spent[0] >= spec["max_usd"]:
            print(f"stopped at the ${spec['max_usd']} cap")
            return


# ---------------------------------------------------------------- reading results


def read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.open() if line.strip()]
    latest = {}
    for r in rows:  # a later ok record supersedes an earlier error
        if r["id"] not in latest or r.get("status") == "ok":
            latest[r["id"]] = r
    return list(latest.values())


def arm_results(arm: str) -> dict[str, list[dict[str, Any]]]:
    return {
        name: read(OUTPUT / arm / name / "predictions.jsonl")
        for name in ("calibration", "framing", "liars")
    }


def margin(r: dict[str, Any]) -> float:
    return r["logit_1"] - r["logit_0"]


def ok(rows):
    return [r for r in rows if r.get("status") == "ok"]


def v03_student() -> dict[str, list[dict[str, Any]]]:
    """Gleipnir 4B v0.3 with the student prompt, from scores we already have.

    Framing and liar originals are the lab_v03 PEFT H100 scores, renamed to
    this experiment's ids; calibration honest rows are lab_v03 `ood_h100`;
    harmful calibration rows and liar swaps come from `v03-merged` (used only
    if parity passed).
    """
    mapping = {
        ("framing", original): condition
        for condition, (name, original) in CONFIG["framing_conditions"].items()
        if name == "framing"
    }
    mapping.update(
        {
            ("framing_v2", original): condition
            for condition, (name, original) in CONFIG["framing_conditions"].items()
            if name == "framing_v2"
        }
    )
    lab = read(LAB / "lab" / "predictions.jsonl")
    framing = []
    for name in ("framing", "framing_v2"):
        for r in read(LAB / name / "predictions.jsonl"):
            condition = mapping.get((name, r["group"]))
            if condition:
                framing.append(
                    {**r, "id": f"{r['parent_id']}::{condition}", "group": condition}
                )
    readers = {r["parent_id"] for r in framing}
    for r in lab:
        if r["id"] in readers:
            framing.append(
                {**r, "id": f"{r['id']}::none", "group": "none", "parent_id": r["id"]}
            )
    calibration_ids = {r["id"]: r for r in load_set("calibration", "student")}
    merged = arm_results("v03-merged")
    honest = [
        {**r, "group": r["group"]}
        for r in read(LAB / "ood_h100" / "predictions.jsonl")
        if r["id"] in calibration_ids and r["label"] == 0
    ]
    harmful = [r for r in merged["calibration"] if r["label"] == 1]
    liars = [r for r in lab if r["group"] in CONFIG["liar_groups"]]
    liars += [r for r in merged["liars"] if "swap" in r["group"]]
    return {"calibration": honest + harmful, "framing": framing, "liars": liars}


# ---------------------------------------------------------------- parity (0b)


def parity(args: argparse.Namespace) -> None:
    import numpy as np

    merged = arm_results("v03-merged")
    peft = {r["id"]: r for r in ok(read(LAB / "ood_h100" / "predictions.jsonl"))}
    peft.update({r["id"]: r for r in ok(read(LAB / "lab" / "predictions.jsonl"))})
    # Honest calibration rows and the 72 unswapped liar/control rows have PEFT
    # scores from the lab_v03 H100 runs; harmful rows and swaps do not.
    pairs = [
        (r, peft[r["id"]])
        for r in ok(merged["calibration"] + merged["liars"])
        if r["id"] in peft
    ]
    a = np.array([r["score"] for r, _ in pairs])
    b = np.array([p["score"] for _, p in pairs])
    calib = [
        i for i, (r, _) in enumerate(pairs) if r["group"] not in CONFIG["liar_groups"]
    ]
    line_m = threshold_at_fpr(list(a[calib]), 0.01)
    line_p = threshold_at_fpr(list(b[calib]), 0.01)
    rules = CONFIG["parity"]
    result = {
        "pairs": len(pairs),
        "calibration_honest_pairs": len(calib),
        "mae": float(np.mean(np.abs(a - b))),
        "max_abs": float(np.max(np.abs(a - b))),
        "pearson": float(np.corrcoef(a, b)[0, 1]),
        "line_1pct_merged": line_m,
        "line_1pct_peft": line_p,
        "flag_agreement_0755": float(np.mean((a > 0.7549) == (b > 0.7549))),
    }
    result["passed"] = bool(
        result["mae"] <= rules["max_mae"]
        and result["pearson"] >= rules["min_pearson"]
        and abs(line_m - line_p) <= rules["max_line_shift_1pct"]
        and result["flag_agreement_0755"] >= rules["min_flag_agreement_0755"]
    )
    (OUTPUT / "parity.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


# ---------------------------------------------------------------- report


def cluster_interval(rows, statistic, seed: int = 0, samples: int = 2000):
    clusters = defaultdict(list)
    for r in rows:
        clusters[r.get("task_id") or r["id"]].append(r)
    keys = sorted(clusters)
    rng = random.Random(seed)
    values = sorted(
        statistic([r for k in rng.choices(keys, k=len(keys)) for r in clusters[k]])
        for _ in range(samples)
    )
    return [values[int(0.025 * samples)], values[int(0.975 * samples) - 1]]


def lines_for(calibration) -> dict[str, float]:
    honest = [r["score"] for r in ok(calibration) if r["label"] == 0]
    return {f"{f:.0%}": threshold_at_fpr(honest, f) for f in CONFIG["fprs"]}


def detection(calibration, lines) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score

    rows = ok(calibration)
    labels = [r["label"] for r in rows]
    harmful = [r["score"] for r in rows if r["label"] == 1]
    result = {"honest": labels.count(0), "harmful": labels.count(1)}
    if len(set(labels)) == 2:
        result["auroc"] = float(roc_auc_score(labels, [r["score"] for r in rows]))
    result["recall"] = (
        {k: sum(s > t for s in harmful) / len(harmful) for k, t in lines.items()}
        if harmful
        else {}
    )
    return result


def framing_table(framing, lines) -> dict[str, Any]:
    rows = ok(framing)
    base = {r["parent_id"]: r for r in rows if r["group"] == "none"}
    table = {}
    for condition in sorted({r["group"] for r in rows}):
        group = [r for r in rows if r["group"] == condition and r["parent_id"] in base]
        entry = {
            "readers": len(group),
            "median_score": median(r["score"] for r in group),
        }
        for key in ("1%", "2%", "5%"):
            t = lines[key]
            entry[f"flagged_{key}"] = sum(r["score"] > t for r in group) / len(group)
        entry["flagged_1%_ci95"] = cluster_interval(
            group, lambda xs, t=lines["1%"]: sum(x["score"] > t for x in xs) / len(xs)
        )
        if condition != "none":
            deltas = [margin(r) - margin(base[r["parent_id"]]) for r in group]
            entry["median_margin_change"] = median(deltas)
            entry["share_margin_up"] = sum(d > 0 for d in deltas) / len(deltas)
            entry["newly_flagged_1%"] = sum(
                r["score"] > lines["1%"] >= base[r["parent_id"]]["score"] for r in group
            )
        table[condition] = entry
    return table


def liar_table(liars, lines) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score

    rows = ok(liars)
    by = defaultdict(list)
    for r in rows:
        by[r["group"]].append(r)
    result = {
        g: {
            "rows": len(v),
            "median_score": median(r["score"] for r in v),
            "flagged_1%": sum(r["score"] > lines["1%"] for r in v),
            "flagged_5%": sum(r["score"] > lines["5%"] for r in v),
        }
        for g, v in sorted(by.items())
    }
    controls = by["honest_recruiter"] + by["assigned_deceiver_no_lie"]
    for g in ("verified_liar", "liar_swap_own", "liar_swap_all"):
        if by[g] and controls:
            labels = [1] * len(by[g]) + [0] * len(controls)
            result[g]["auroc_vs_recruiter_controls"] = float(
                roc_auc_score(labels, [r["score"] for r in by[g] + controls])
            )
    originals = {r["id"]: r for r in by["verified_liar"]}
    for g in ("liar_swap_own", "liar_swap_all"):
        deltas = [
            margin(r) - margin(originals[r["parent_id"]])
            for r in by[g]
            if r.get("parent_id") in originals
        ]
        if deltas:
            result[g]["median_margin_change_vs_original"] = median(deltas)
            result[g]["share_margin_down"] = sum(d < 0 for d in deltas) / len(deltas)
    return result


def arm_report(results) -> dict[str, Any]:
    if not ok(results["calibration"]):
        return {"status": "no calibration scores"}
    lines = lines_for(results["calibration"])
    out = {"lines": lines, "detection": detection(results["calibration"], lines)}
    if ok(results["framing"]):
        out["framing"] = framing_table(results["framing"], lines)
    if ok(results["liars"]):
        out["liars"] = liar_table(results["liars"], lines)
    statuses = defaultdict(int)
    for rows in results.values():
        for r in rows:
            statuses[r.get("status")] += 1
    out["statuses"] = dict(statuses)
    return out


def layer_two(student, arms) -> dict[str, Any]:
    """Of subset readers the 4B (student) flags at its 1% line, who else flags."""
    subset = {
        r["metadata"]["parent_id"]
        for r in load_set("framing", "student")
        if r["metadata"]["subset"]
    }
    s_lines = lines_for(student["calibration"])
    first = {
        (r["parent_id"], r["group"]): r["score"] > s_lines["1%"]
        for r in ok(student["framing"])
        if r["parent_id"] in subset
    }
    result = {}
    for arm in ("27b-teacher", "kimi-teacher"):
        res = arms.get(arm)
        if not res or not ok(res["calibration"]):
            continue
        lines = lines_for(res["calibration"])
        second = {
            (r["parent_id"], r["group"]): r["score"] > lines["1%"]
            for r in ok(res["framing"])
        }
        entry = {}
        for condition in CONFIG["subset"]["conditions"]:
            keys = [k for k in second if k[1] == condition and k in first]
            flagged_first = [k for k in keys if first[k]]
            entry[condition] = {
                "readers": len(keys),
                "flagged_by_4b": len(flagged_first),
                "also_flagged": sum(second[k] for k in flagged_first),
                "flagged_by_this_arm": sum(second[k] for k in keys),
            }
        result[arm] = entry
    return result


def step_0a(student) -> dict[str, Any]:
    """Recall-first lines for the 4B student, on scores we already have."""
    honest = [
        r["score"]
        for r in ok(read(LAB / "ood_h100" / "predictions.jsonl"))
        if r["label"] == 0
    ]
    lines = {f"{f:.0%}": threshold_at_fpr(honest, f) for f in (0.01, 0.05, 0.10, 0.20)}
    lab = ok(read(LAB / "lab" / "predictions.jsonl"))
    framing = ok(read(LAB / "framing" / "predictions.jsonl")) + ok(
        read(LAB / "framing_v2" / "predictions.jsonl")
    )
    harmful = [r["score"] for r in ok(student["calibration"]) if r["label"] == 1]
    groups = defaultdict(list)
    for r in lab:
        groups[f"lab:{r['group']}"].append(r["score"])
    for r in framing:
        groups[f"framing:{r['group']}"].append(r["score"])
    table = {
        name: {k: sum(s > t for s in scores) / len(scores) for k, t in lines.items()}
        | {"rows": len(scores)}
        for name, scores in sorted(groups.items())
    }
    recall = (
        {k: sum(s > t for s in harmful) / len(harmful) for k, t in lines.items()}
        if harmful
        else {}
    )
    liars = sorted(
        (r["score"] for r in lab if r["group"] == "verified_liar"), reverse=True
    )
    plain = [r["score"] for r in lab if r["group"] == "plain"]
    catch = []
    for k in range(1, len(liars) + 1):
        t = liars[k - 1] - 1e-9
        catch.append(
            {
                "liars_caught": k,
                "line": t,
                "plain_workers_escalated": sum(s > t for s in plain) / len(plain),
                "benchmark_fpr": sum(s > t for s in honest) / len(honest),
                "benchmark_recall_on_calibration_harmful": (
                    sum(s > t for s in harmful) / len(harmful) if harmful else None
                ),
            }
        )
    return {
        "lines": lines,
        "benchmark_recall": recall,
        "escalation_share": table,
        "line_to_catch_liars": catch,
    }


def report(args: argparse.Namespace) -> None:
    parity_result = (
        json.loads((OUTPUT / "parity.json").read_text())
        if (OUTPUT / "parity.json").exists()
        else None
    )
    student = v03_student()
    arms = {arm: arm_results(arm) for arm in CONFIG["arms"] if arm != "v03-merged"}
    result = {
        "parity_0b": parity_result,
        "step_0a": step_0a(student),
        "arms": {"v03-student": arm_report(student)}
        | {a: arm_report(r) for a, r in arms.items()},
        "layer_two": layer_two(student, arms),
    }
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2)[: args.chars])


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("smoke", "score"):
        command = commands.add_parser(name)
        command.add_argument("--arm", required=True, choices=sorted(CONFIG["arms"]))
        command.add_argument("--monitor")
        command.add_argument("--device", default="cuda")
    commands.choices["score"].add_argument("--sets", nargs="+")
    kimi_command = commands.add_parser("kimi")
    kimi_command.add_argument("--sets", nargs="+")
    kimi_command.add_argument("--limit", type=int)
    commands.add_parser("parity")
    report_command = commands.add_parser("report")
    report_command.add_argument("--chars", type=int, default=12000)
    args = parser.parse_args()
    {"smoke": smoke, "score": score, "kimi": kimi, "parity": parity, "report": report}[
        args.command
    ](args)


if __name__ == "__main__":
    main()
