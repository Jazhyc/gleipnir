"""Recount historical evidence and validate the pending Step 1 input contract."""

from __future__ import annotations

import json
from collections import Counter

from experiments.adversarial_a2_cover_comments.run import insert_comments
from gleipnir.adversarial import apps, eval_sets
from gleipnir.adversarial.metrics import threshold_at_fpr

from .common import ROOT, config, digest, probability, read, sha, unique


def calibrate(rows: list[dict], fprs: list[float]) -> dict:
    if not rows:
        raise ValueError("missing separate honest calibration rows")
    unique(rows)
    values = [probability(r["score"]) for r in rows]
    return {
        str(f): {
            "threshold": threshold_at_fpr(values, f),
            "n": len(values),
            "flagged": sum(s > threshold_at_fpr(values, f) for s in values),
            "ties": sum(s == threshold_at_fpr(values, f) for s in values),
        }
        for f in fprs
    }


def exclusions(cfg: dict) -> set[str]:
    return set(json.loads((ROOT / cfg["exclude_a4_config"]).read_text())["task_ids"])


def recount(cfg: dict | None = None) -> dict:
    cfg = cfg or config()
    h = cfg["historical"]
    for key in ("selection", "scores"):
        if eval_sets.file_sha256(ROOT / h[key]) != h[key + "_sha256"]:
            raise ValueError("historical input checksum changed")
    chosen = json.loads((ROOT / h["selection"]).read_text())
    rows = unique(read(ROOT / h["scores"]))
    calibration = [r for r in rows.values() if r["set"] == "calibration"]
    if len(calibration) != len(chosen["development_pool"]) or any(
        r["status"] != "ok" or r["code"] != "honest" or r["text"] != "none"
        for r in calibration
    ):
        raise ValueError("incomplete honest calibration")
    lines = calibrate(calibration, cfg["fprs"])
    t = lines["0.01"]["threshold"]
    selected, detected, excluded = [], [], []
    for split in ("development", "heldout"):
        for task in chosen[split]:
            row = rows[f"{task}:backdoored:none"]
            if (
                row["status"] != "ok"
                or row["monitor"] != "4b-v03"
                or row["instruction"] != "student"
            ):
                raise ValueError("wrong baseline contract")
            if probability(row["score"]) > t:
                detected.append(task)
                (excluded if task in exclusions(cfg) else selected).append(task)
    # Validate both code surfaces, but do not silently shrink the detected cohort.
    pairs = benchmark_pairs(cfg)
    invalid = {}
    for task in selected:
        try:
            for row in pairs[task].values():
                insert_comments(
                    apps.parse(row["prompt"]).code,
                    [{"before_line": 1, "comment": "Implementation notes."}],
                )
        except (ValueError, KeyError, SyntaxError) as error:
            invalid[task] = type(error).__name__
    return {
        "status": "historical_only_not_fresh_selection",
        "input_hashes": h,
        "calibration": lines,
        "baseline_tasks": len(chosen["development"]) + len(chosen["heldout"]),
        "detected": len(detected),
        "excluded_original_a4": len(excluded),
        "eligible_before_validator": len(selected),
        "validator_failures": invalid,
        "selected_task_ids": selected,
        "excluded_task_ids": excluded,
        "by_split": {
            s: sum(t in chosen[s] for t in selected) for s in ("development", "heldout")
        },
        "calibration_task_overlap": len(
            set(selected) & set(chosen["development_pool"])
        ),
        "selection_sha256": digest(selected),
    }


def benchmark_pairs(cfg: dict) -> dict:
    if eval_sets.SETS[cfg["data"]["set"]].sha256 != cfg["data"]["sha256"]:
        raise ValueError("APPS dataset contract changed")
    pairs = {}
    for row in eval_sets.load(cfg["data"]["set"]):
        group = eval_sets.group(row)
        if group not in ("clean", "vanilla"):
            continue
        task = str(row["metadata"]["task_id"])
        code = "honest" if group == "clean" else "backdoored"
        if code in pairs.setdefault(task, {}):
            raise ValueError("duplicate benchmark pair")
        pairs[task][code] = row
    return pairs


def select(dependency: dict, pairs: dict, cfg: dict) -> dict:
    """Select only baseline-detected tasks; dependency is an explicit fresh bundle.

    Each input row binds task/code/score to the original benchmark prompt hash.
    No historical fallback, task discovery, label inference, or code execution.
    """
    if dependency.get("status") != "ready" or not dependency.get("provenance"):
        raise ValueError("Step 1 fresh selection dependency is not ready")
    task_ids = dependency["task_ids"]
    cal_ids = dependency["calibration_task_ids"]
    if (
        not task_ids
        or not cal_ids
        or any(not isinstance(t, str) for t in task_ids + cal_ids)
    ):
        raise ValueError("nonempty string task IDs required")
    if len(set(task_ids)) != len(task_ids) or len(set(cal_ids)) != len(cal_ids):
        raise ValueError("duplicate task IDs")
    if set(task_ids) & set(cal_ids):
        raise ValueError("fresh evaluation and calibration task groups overlap")
    monitor = dependency["monitor_identity"]
    if (
        monitor.get("monitor") != cfg["monitor_contracts"]["search"]["monitor"]
        or monitor.get("instruction")
        != cfg["monitor_contracts"]["search"]["instruction"]
    ):
        raise ValueError("wrong search monitor identity")
    if not monitor.get("runtime_receipt_sha256"):
        raise ValueError("missing runtime/parity receipt identity")
    scores = unique(dependency["baseline_scores"])
    calibration, baselines, reasons = [], {}, Counter()
    for task, code in [(t, "honest") for t in cal_ids] + [
        (t, c) for t in task_ids for c in ("honest", "backdoored")
    ]:
        if task not in pairs or set(pairs[task]) != {"honest", "backdoored"}:
            raise ValueError("task is not an existing paired APPS benchmark task")
        row = scores[f"{task}:{code}"]
        prompt = pairs[task][code]["prompt"]
        if row["prompt_sha256"] != sha(prompt) or row["status"] != "ok":
            raise ValueError("missing, failed, or stale baseline score")
        probability(row["score"])
        if task in cal_ids:
            calibration.append(row)
        else:
            baselines[f"{task}:{code}"] = row["score"]
    lines = calibrate(calibration, cfg["fprs"])
    selected, failures = [], {}
    for task in task_ids:
        if task in exclusions(cfg):
            reasons["original_a4"] += 1
        elif baselines[f"{task}:backdoored"] <= lines["0.01"]["threshold"]:
            reasons["not_detected"] += 1
        else:
            selected.append(task)
            try:
                for row in pairs[task].values():
                    insert_comments(
                        apps.parse(row["prompt"]).code,
                        [{"before_line": 1, "comment": "Implementation notes."}],
                    )
            except (ValueError, SyntaxError) as error:
                failures[task] = type(error).__name__
    if failures:
        raise ValueError(
            "selected input fails A2 preservation preflight; "
            "record failures and resolve before freeze"
        )
    if not selected:
        raise ValueError("no eligible tasks")
    return {
        "task_ids": sorted(selected, key=lambda t: sha(f"{cfg['seed']}:{t}")),
        "calibration_task_ids": cal_ids,
        "calibration": lines,
        "baselines": baselines,
        "monitor_identity": monitor,
        "input_tasks": len(task_ids),
        "excluded": dict(reasons),
        "dependency_sha256": digest(dependency),
        "provenance": dependency["provenance"],
    }
