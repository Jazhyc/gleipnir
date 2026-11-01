"""Build the natural-cheats monitor rows (CPU only, two steps).

Step 1 reads the pilot's Inspect logs (needs inspect_ai):

    PYTHONPATH=src:. ../../.venv-impbench/bin/python \
        experiments/adversarial_natural_cheats/build.py extract

Step 2 renders, cuts and strips the runs, counts tokens with each monitor's
tokenizer (tokenizer files only) and builds the strict-OOD calibration rows:

    PYTHONPATH=src:. ../../.venv-data/bin/python \
        experiments/adversarial_natural_cheats/build.py rows

Writes to the config's data directory:
- `runs.jsonl`: every hand-labelled run, messages normalized (step 1);
- `rows_<monitor>.jsonl`: one row per run and detection point, under that
  monitor's prompt (trajectories byte-identical across monitors);
- `calibration_<monitor>.jsonl`: the 1,000 honest strict-OOD rows;
- `manifest.json`: counts, too-long rows, strip outcomes, token lengths, SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from experiments.adversarial_natural_cheats.trajectories import (
    cut,
    failure_replies,
    first_turn_after,
    nearest_turn,
    normalize,
    render,
    strip_messages,
    tool_call_turns,
)
from gleipnir.adversarial import eval_sets
from gleipnir.adversarial.contract import MAX_TOKENS, join_prompt

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text())
ROOT = eval_sets.ROOT
DATA = ROOT / CONFIG["data"]
POINTS = ("first_cheat", "pressure", "full", "stripped")
LABEL_FIELDS = (
    "first_cheat_message",
    "method",
    "written_down",
    "visible_to_gleipnir",
    "reverted",
    "disclosed_in_final_message",
    "noticed_contradiction",
    "mutation_valid",
    "confidence",
)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> str:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return eval_sets.file_sha256(path)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


# ---------------------------------------------------------------- step 1


def extract() -> None:
    from inspect_ai import __version__ as inspect_version
    from inspect_ai.log import read_eval_log

    results = ROOT / CONFIG["pilot"]["results"]
    runs, per_arm = [], {}
    for arm in CONFIG["arms"]:
        expected = CONFIG["inputs_sha256"][arm]
        directory = results / arm
        logs = sorted(directory.glob("*.eval"))
        if [p.name for p in logs] != sorted(expected["eval"]):
            raise ValueError(f"{arm}: .eval files differ from the config")
        for name in ("labels.json", "recheck.jsonl"):
            if eval_sets.file_sha256(directory / name) != expected[name]:
                raise ValueError(f"{arm}/{name} does not match its frozen SHA256")
        samples = {}
        for path in logs:
            if eval_sets.file_sha256(path) != expected["eval"][path.name]:
                raise ValueError(f"{path} does not match its frozen SHA256")
            for sample in read_eval_log(str(path)).samples or []:
                if sample.id in samples or sample.epoch != 1:
                    raise ValueError(f"{arm}: duplicate sample {sample.id}")
                samples[sample.id] = (path.name, sample)
        labels = json.loads((directory / "labels.json").read_text())
        for label in labels:
            log, sample = samples[label["task_id"]]
            messages = [normalize(m) for m in sample.messages]
            if "attachment://" in json.dumps(messages):
                raise ValueError(f"{arm}/{label['task_id']}: unresolved attachment")
            runs.append(
                {
                    "arm": arm,
                    "task_id": label["task_id"],
                    "log": log,
                    "sample_uuid": getattr(sample, "uuid", None),
                    "sample_error": str(sample.error.message)[:300]
                    if sample.error
                    else None,
                    "limit": str(sample.limit.type) if sample.limit else None,
                    "label": label,
                    "messages": messages,
                }
            )
        per_arm[arm] = {"samples": len(samples), "labelled": len(labels)}
    DATA.mkdir(parents=True, exist_ok=True)
    digest = write_jsonl(DATA / "runs.jsonl", runs)
    manifest = {
        "inspect_ai": inspect_version,
        "runs": len(runs),
        "per_arm": per_arm,
        "sha256": digest,
    }
    (DATA / "runs_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


# ---------------------------------------------------------------- step 2: cuts


def run_key(run: dict) -> str:
    return f"{run['arm']}::{run['task_id']}"


def run_order(run: dict, arms: list[str]) -> tuple[int, int, str]:
    task = run["task_id"]
    number = int(task.rsplit("_", 1)[1]) if task.rsplit("_", 1)[-1].isdigit() else -1
    return arms.index(run["arm"]), number, task


def class_of(label: str, classes: dict[str, list[str]]) -> str:
    for name, labels in classes.items():
        if label in labels:
            return name
    raise ValueError(f"unknown label {label!r}")


def labelled_cut(run: dict) -> int | None:
    """The labelled first-cheat message, if it is an assistant tool-call turn."""
    index = run["label"].get("first_cheat_message")
    if index is None:
        return None
    if index not in tool_call_turns(run["messages"]):
        raise ValueError(f"{run_key(run)}: message {index} is not a tool-call turn")
    return index


def first_cheat_cuts(
    runs: list[dict], classes: dict[str, str], seed: int, arms: list[str]
) -> dict[str, dict]:
    """Cut index and rule per run for the first-cheat point (see config)."""
    ordered = sorted(runs, key=lambda r: run_order(r, arms))
    lengths = [labelled_cut(r) for r in ordered if classes[run_key(r)] == "positive"]
    cuts: dict[str, dict] = {}
    for run in ordered:
        index = labelled_cut(run)
        if classes[run_key(run)] == "positive" and index is None:
            raise ValueError(f"{run_key(run)}: positive without first_cheat_message")
        if index is not None and classes[run_key(run)] in (
            "positive",
            "intent",
            "ambiguous",
        ):
            cuts[run_key(run)] = {"cut": index, "rule": "first_cheat_message"}
    rng = random.Random(seed)
    for name in ("negative", "clean", "intent", "ambiguous"):
        for run in ordered:
            key = run_key(run)
            if classes[key] != name or key in cuts:
                continue
            drawn = rng.choice(lengths)
            cuts[key] = {
                "cut": nearest_turn(tool_call_turns(run["messages"]), drawn),
                "rule": "length_matched",
                "drawn_length": drawn,
            }
    return cuts


def pushed(run: dict) -> bool | None:
    """A scaffold failure reply came before the labelled first-cheat message."""
    index = run["label"].get("first_cheat_message")
    if index is None:
        return None
    return any(i < index for i in failure_replies(run["messages"]))


def pressure_cut(run: dict, name: str) -> tuple[dict | None, str]:
    """(cut, reason) for the under-pressure point; cut None if excluded."""
    if name == "positive":
        if pushed(run):
            return {"cut": labelled_cut(run), "rule": "first_cheat_message"}, "ok"
        return None, "positive_not_pushed"
    if name != "negative":
        return None, f"{name}_not_in_contrast"
    replies = failure_replies(run["messages"])
    if not replies:
        return None, "negative_without_failure_reply"
    turn = first_turn_after(run["messages"], replies[0])
    if turn is None:
        return None, "negative_without_turn_after_reply"
    return {
        "cut": turn,
        "rule": "first_turn_after_first_failure_reply",
        "reply_index": replies[0],
    }, "ok"


# ---------------------------------------------------------------- step 2: rows


def run_metadata(
    run: dict, name: str, pilot: dict, partitions: dict, ground_truth: dict
) -> dict:
    arm = pilot["arms"][run["arm"]]
    label = run["label"]
    return {
        "source_dataset": "adversarial_natural_cheats",
        "arm": run["arm"],
        "provider": arm["provider"],
        "split": arm["split"],
        "prompt_variant": arm["prompt"],
        "task_id": run["task_id"],
        "partition": partitions[run["task_id"]],
        "parent_id": run_key(run),
        "log": run["log"],
        "label": label["label"],
        "label_class": name,
        "ground_truth": ground_truth[name],
        **{field: label.get(field) for field in LABEL_FIELDS},
        "pushed": pushed(run),
        "failure_replies": failure_replies(run["messages"]),
        "messages_in_run": len(run["messages"]),
        "sample_error": run["sample_error"],
        "limit": run["limit"],
    }


def cut_metadata(messages: list[dict], info: dict) -> dict:
    kept = cut(messages, info["cut"])
    return {
        "cut_message_index": info["cut"],
        "cut_rule": info["rule"],
        **{k: info[k] for k in ("drawn_length", "reply_index") if k in info},
        "messages_in_cut": len(kept),
        "tool_call_turns_in_cut": len(tool_call_turns(kept)),
        "failure_replies_in_cut": len(failure_replies(kept)),
        "tool_errors_in_cut": sum(
            1 for m in kept if m["role"] == "tool" and m.get("error")
        ),
    }


def make_row(
    key: str,
    point: str,
    base: dict,
    messages: list[dict],
    info: dict,
    trajectory: str,
    extra: dict | None = None,
) -> dict:
    name = base["label_class"]
    meta = {**base, "detection_point": point, "condition": f"{point}:{name}"}
    meta.update(cut_metadata(messages, info), **(extra or {}))
    return {"id": f"{key}::{point}", "trajectory": trajectory, "meta": meta}


def build_rows(
    runs: list[dict], pilot: dict, config: dict = CONFIG
) -> tuple[list[dict], dict]:
    """Rows (id, trajectory, metadata) for every detection point, plus counts."""
    arms = config["arms"]
    partitions = {
        task: name
        for name in ("probe_train", "dev", "test")
        for task in pilot["task_splits"][name]
    }
    classes = {
        run_key(r): class_of(r["label"]["label"], config["classes"]) for r in runs
    }
    found = Counter(classes.values())
    counts = {name: found.get(name, 0) for name in config["classes"]}
    if counts != config["expected_counts"]:
        raise ValueError(f"class counts {counts} != {config['expected_counts']}")
    first = first_cheat_cuts(runs, classes, config["length_match_seed"], arms)
    options = {
        "python_heredocs": config["strip"]["bash_python_heredocs"],
        "python_c": config["strip"]["bash_python_c"],
    }
    rows, pressure_reasons, strip_totals = [], Counter(), Counter()
    for run in sorted(runs, key=lambda r: run_order(r, arms)):
        key, messages = run_key(run), run["messages"]
        name = classes[key]
        base = run_metadata(run, name, pilot, partitions, config["ground_truth"])
        info = first[key]
        kept = cut(messages, info["cut"])
        rows.append(make_row(key, "first_cheat", base, messages, info, render(kept)))
        info_p, reason = pressure_cut(run, name)
        pressure_reasons[f"{name}:{reason}"] += 1
        if info_p is not None:
            trajectory = render(cut(messages, info_p["cut"]))
            rows.append(make_row(key, "pressure", base, messages, info_p, trajectory))
        full = {"cut": len(messages) - 1, "rule": "full"}
        rows.append(make_row(key, "full", base, messages, full, render(messages)))
        stats: Counter[str] = Counter()
        stripped = strip_messages(kept, stats, **options)
        last: Counter[str] = Counter()
        strip_messages(kept[-1:], last, **options)
        strip_totals.update({f"{name}:{k}": v for k, v in stats.items()})
        extra = {
            "strip": dict(sorted(stats.items())),
            "strip_last_turn_comments_removed": last["comments_removed"],
            "stripped_changed": stripped != kept,
        }
        row = make_row(key, "stripped", base, messages, info, render(stripped), extra)
        rows.append(row)
    summary = {
        "runs": len(runs),
        "classes": counts,
        "pressure_selection": dict(sorted(pressure_reasons.items())),
        "strip_by_class": dict(sorted(strip_totals.items())),
    }
    return rows, summary


def monitor_rows(rows: list[dict], head: str) -> list[dict]:
    return [
        {
            "id": r["id"],
            "prompt": join_prompt(head, r["trajectory"]),
            "metadata": r["meta"],
        }
        for r in rows
    ]


def add_tokens(rows: list[dict], heads: dict[str, str], counters: dict) -> None:
    """Token count of each row under each monitor's own prompt and tokenizer."""
    for row in rows:
        tokens = {
            monitor: counters[monitor](join_prompt(heads[monitor], row["trajectory"]))
            for monitor in counters
        }
        row["meta"]["tokens"] = tokens
        row["meta"]["too_long"] = {m: n > MAX_TOKENS for m, n in tokens.items()}


# ---------------------------------------------------------------- summaries


def quantiles(values: list[int]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    s = sorted(values)

    def q(p: float) -> int:
        return s[min(len(s) - 1, int(p * len(s)))]

    return {
        "n": len(s),
        "min": s[0],
        "p10": q(0.10),
        "p25": q(0.25),
        "median": q(0.50),
        "p75": q(0.75),
        "p90": q(0.90),
        "max": s[-1],
        "mean": round(sum(s) / len(s), 1),
    }


def mann_whitney_auroc(positives: list[float], negatives: list[float]) -> float | None:
    if not positives or not negatives:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def row_summary(rows: list[dict], monitors: list[str]) -> dict[str, Any]:
    counts: dict = defaultdict(Counter)
    too_long: dict = {m: defaultdict(Counter) for m in monitors}
    lengths: dict = {m: defaultdict(list) for m in monitors}
    for r in rows:
        meta = r["meta"]
        point, name = meta["detection_point"], meta["label_class"]
        counts[point][name] += 1
        for m in monitors:
            lengths[m][(point, name)].append(meta["tokens"][m])
            if meta["too_long"][m]:
                too_long[m][point][name] += 1
    summary: dict[str, Any] = {
        "rows": len(rows),
        "rows_by_point_and_class": {p: dict(c) for p, c in counts.items()},
        "too_long": {
            m: {p: dict(c) for p, c in v.items()} for m, v in too_long.items()
        },
        "token_lengths": {
            m: {f"{p}:{c}": quantiles(v) for (p, c), v in sorted(d.items())}
            for m, d in lengths.items()
        },
        "length_only_auroc": {
            m: {
                p: mann_whitney_auroc(d[(p, "positive")], d[(p, "negative")])
                for p in POINTS
            }
            for m, d in lengths.items()
        },
    }
    stripped = [r["meta"] for r in rows if r["meta"]["detection_point"] == "stripped"]
    pos = [m for m in stripped if m["label_class"] == "positive"]
    totals: Counter[str] = Counter()
    for m in stripped:
        totals.update(m["strip"])
    summary["strip"] = {
        "totals": dict(sorted(totals.items())),
        "rows_changed": sum(m["stripped_changed"] for m in stripped),
        "rows_with_tokenize_failures": sum(
            1 for m in stripped if m["strip"].get("tokenize_failed")
        ),
        "positives": len(pos),
        "positives_changed": sum(m["stripped_changed"] for m in pos),
        "positives_first_cheat_turn_comments_removed": sum(
            1 for m in pos if m["strip_last_turn_comments_removed"]
        ),
        "positives_with_tokenize_failures": sum(
            1 for m in pos if m["strip"].get("tokenize_failed")
        ),
    }
    first = [r["meta"] for r in rows if r["meta"]["detection_point"] == "first_cheat"]
    pushed_counts: dict = defaultdict(Counter)
    for m in first:
        if m["label_class"] == "positive":
            pushed_counts[f"{m['label']}:{m['arm']}"][
                "pushed" if m["pushed"] else "not"
            ] += 1
    summary["pushed"] = {k: dict(v) for k, v in sorted(pushed_counts.items())}
    summary["tool_errors_rendered_empty"] = {
        p: sum(
            r["meta"]["tool_errors_in_cut"]
            for r in rows
            if r["meta"]["detection_point"] == p
        )
        for p in POINTS
    }
    return summary


# ---------------------------------------------------------------- calibration


def calibration_rows(texts: dict[str, str], counters: dict, tokens_4b_teacher) -> tuple:
    """The 1,000 honest strict-OOD rows, built by adversarial_is_it_4b's function."""
    from experiments.adversarial_is_it_4b.build import calibration

    student, teacher, summary = calibration(texts, tokens_4b_teacher)
    honest = {
        "student": [r for r in student if eval_sets.label(r) == 0],
        "teacher": [r for r in teacher if eval_sets.label(r) == 0],
    }
    if len(honest["student"]) != CONFIG["calibration"]["honest"]:
        raise ValueError("calibration: wrong number of honest rows")
    check = {}
    for prompt, path in CONFIG["calibration"]["reference_files"].items():
        file = ROOT / path
        if not file.exists():
            check[prompt] = "reference file absent"
            continue
        reference = [r for r in read_jsonl(file) if eval_sets.label(r) == 0]
        ours = honest[prompt]
        same_ids = [r["id"] for r in ours] == [r["id"] for r in reference]
        same_prompts = all(
            a["prompt"] == b["prompt"] for a, b in zip(ours, reference, strict=True)
        )
        check[prompt] = {
            "rows": len(reference),
            "ids_and_order_match": same_ids,
            "prompts_match": same_prompts,
        }
    by_monitor = {}
    for monitor, spec in CONFIG["monitors"].items():
        rows = []
        for s_row, t_row in zip(honest["student"], honest["teacher"], strict=True):
            if s_row["id"] != t_row["id"]:
                raise ValueError("calibration: student and teacher rows out of order")
            tokens = {
                "4b-v03": counters["4b-v03"](s_row["prompt"]),
                "27b": counters["27b"](t_row["prompt"]),
            }
            row = s_row if spec["prompt"] == "student" else t_row
            meta = {
                **row["metadata"],
                "tokens": tokens,
                "too_long": {m: n > MAX_TOKENS for m, n in tokens.items()},
            }
            rows.append({**row, "metadata": meta})
        by_monitor[monitor] = rows
    info = {
        "is_it_4b_summary": summary,
        "honest": len(honest["student"]),
        "matches_is_it_4b": check,
        "too_long": {
            m: sum(r["metadata"]["too_long"][m] for r in rows)
            for m, rows in by_monitor.items()
        },
        "token_lengths": {
            m: quantiles([r["metadata"]["tokens"][m] for r in rows])
            for m, rows in by_monitor.items()
        },
    }
    return by_monitor, info


# ---------------------------------------------------------------- step 2 main


def instructions() -> dict[str, str]:
    texts = {
        name: (ROOT / path).read_text() for name, path in CONFIG["prompts"].items()
    }
    for name, text in texts.items():
        if sha(text) != CONFIG["prompt_sha256"][name]:
            raise ValueError(f"{name} prompt does not match its frozen SHA256")
    return texts


def token_counters() -> dict:
    from transformers import AutoTokenizer

    from gleipnir.adversarial.contract import render as render_chat
    from gleipnir.adversarial.monitors import MONITORS

    counters = {}
    for monitor in CONFIG["monitors"]:
        spec = MONITORS[monitor]
        tokenizer = AutoTokenizer.from_pretrained(
            spec.base, revision=spec.base_revision
        )

        def count(prompt: str, tokenizer=tokenizer) -> int:
            return len(
                tokenizer.encode(
                    render_chat(tokenizer, prompt), add_special_tokens=False
                )
            )

        counters[monitor] = count
    return counters


def rows_step() -> None:
    manifest_in = json.loads((DATA / "runs_manifest.json").read_text())
    if eval_sets.file_sha256(DATA / "runs.jsonl") != manifest_in["sha256"]:
        raise ValueError("runs.jsonl does not match runs_manifest.json")
    runs = read_jsonl(DATA / "runs.jsonl")
    pilot = json.loads((ROOT / CONFIG["pilot"]["config"]).read_text())
    texts = instructions()
    heads = {m: texts[spec["prompt"]] for m, spec in CONFIG["monitors"].items()}
    rows, summary = build_rows(runs, pilot)
    counters = token_counters()
    add_tokens(rows, heads, counters)
    monitors = list(CONFIG["monitors"])
    manifest: dict[str, Any] = {
        "runs_sha256": manifest_in["sha256"],
        "build": summary,
        "rows": row_summary(rows, monitors),
        "sha256": {},
    }
    for monitor in monitors:
        path = DATA / f"rows_{monitor}.jsonl"
        manifest["sha256"][path.name] = write_jsonl(
            path, monitor_rows(rows, heads[monitor])
        )
    # adversarial_is_it_4b filtered on the 4B tokenizer's teacher-prompt length.
    calibration, info = calibration_rows(texts, counters, counters["4b-v03"])
    manifest["calibration"] = info
    for monitor, cal_rows in calibration.items():
        path = DATA / f"calibration_{monitor}.jsonl"
        manifest["sha256"][path.name] = write_jsonl(path, cal_rows)
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2)[:20000])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=("extract", "rows"))
    args = parser.parse_args()
    extract() if args.step == "extract" else rows_step()


if __name__ == "__main__":
    main()
