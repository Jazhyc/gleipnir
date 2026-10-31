"""Build the EvilGenie x Escalation monitor rows (CPU only, one step).

The dataset is plain JSON, so no Inspect environment is needed:

    PYTHONPATH=src:. HF_HUB_OFFLINE=1 ../../.venv-data/bin/python \
        experiments/adversarial_evilgenie/build.py rows [--record]
    PYTHONPATH=src:. ../../.venv-data/bin/python \
        experiments/adversarial_evilgenie/build.py hand-check

`rows` checks every input file against config.json, joins each transcript to
its label on `run_key`, renders the full run in the strict-OOD action-only
format (the natural-cheats renderer), counts tokens under each monitor's prompt
and tokenizer (tokenizer files only) and writes to the config's data directory:
- `rows_<monitor>.jsonl`: one row per run, under that monitor's prompt
  (trajectories byte-identical across monitors);
- `manifest.json`: schema survey, counts, too-long rows, token lengths, SHA-256.
`--record` writes the rows' SHA-256 into config.json (score/analyze refuse
rows that do not match it).

`hand-check` prints 10 random positives and 10 random negatives (seed 0) as
rendered trajectory tails plus the dataset's judge explanation, and saves them
to `hand_check.md`.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from experiments.adversarial_natural_cheats.build import (
    add_tokens,
    mann_whitney_auroc,
    monitor_rows,
    quantiles,
    read_jsonl,
    sha,
    write_jsonl,
)
from experiments.adversarial_natural_cheats.trajectories import render
from gleipnir.adversarial import eval_sets
from gleipnir.adversarial.contract import split_prompt

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.json"
CONFIG = json.loads(CONFIG_PATH.read_text())
ROOT = eval_sets.ROOT
DATA = ROOT / CONFIG["data"]
RAW = ROOT / CONFIG["source"]["local_dir"]
CONTINUE_PROMPT = "Please proceed to the next step using your best judgement."
ESCALATION_TOOL = "submit_escalation_report"
HAND_CHECK = {"per_class": 10, "seed": 0, "tail_chars": 1500}


# ---------------------------------------------------------------- inputs


def verify_inputs(raw: Path = RAW, config: dict = CONFIG) -> None:
    """Every file used matches its SHA-256; no transcript file is unlisted."""
    expected = config["inputs_sha256"]
    present = {f"transcripts/{p.name}" for p in (raw / "transcripts").glob("*.jsonl")}
    listed = {name for name in expected if name.startswith("transcripts/")}
    if present != listed:
        raise ValueError(f"transcript files differ from config: {present ^ listed}")
    for name, digest in expected.items():
        if eval_sets.file_sha256(raw / name) != digest:
            raise ValueError(f"{name} does not match its SHA-256 in config.json")


def read_labels(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels = {r["run_key"]: r for r in rows}
    if len(labels) != len(rows):
        raise ValueError("labels: duplicate run_key")
    return labels


def read_transcripts(raw: Path = RAW) -> list[dict[str, Any]]:
    """Every run, with the file and line it came from."""
    runs = []
    for path in sorted((raw / "transcripts").glob("*.jsonl")):
        with path.open() as handle:
            for line_number, line in enumerate(handle, 1):
                run = json.loads(line)
                run["_file"], run["_line"] = path.name, line_number
                runs.append(run)
    keys = Counter(r["run_key"] for r in runs)
    if duplicated := [k for k, n in keys.items() if n > 1]:
        raise ValueError(f"duplicate run_key in transcripts: {duplicated[:3]}")
    return runs


# ---------------------------------------------------------------- messages


def normalize(message: dict[str, Any]) -> dict[str, Any]:
    """One exported message as the natural-cheats renderer's plain dict.

    Content is a string, or (assistant only) a list of `text` and `reasoning`
    parts; reasoning is dropped here and assistant text by the renderer. The
    export has no tool `error` field, so `error` is always None and an errored
    result keeps its stored (empty) content.
    """
    role = message["role"]
    content = message.get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list) and role == "assistant":
        kinds = {part.get("type") for part in content}
        if not kinds <= {"text", "reasoning"}:
            raise ValueError(f"unknown content part types {kinds}")
        text = "\n".join(p.get("text") or "" for p in content if p["type"] == "text")
    else:
        raise ValueError(f"unexpected {role} content {type(content).__name__}")
    out: dict[str, Any] = {"role": role, "text": text}
    if role == "assistant":
        calls = []
        for call in message.get("tool_calls") or []:
            if not isinstance(call.get("arguments"), dict):
                raise ValueError(f"tool call {call.get('id')}: arguments not a dict")
            calls.append({"function": call["function"], "arguments": call["arguments"]})
        out["tool_calls"] = calls
    elif role == "tool":
        out["function"] = message.get("function")
        out["error"] = None
    elif role not in ("system", "user"):
        raise ValueError(f"unknown role {role!r}")
    return out


def run_stats(raw_messages: list[dict], messages: list[dict]) -> dict[str, Any]:
    """Counts describing one run (from the exported and normalized messages)."""
    calls = [c for m in messages if m["role"] == "assistant" for c in m["tool_calls"]]
    answered = {m.get("tool_call_id") for m in raw_messages if m["role"] == "tool"} - {
        None
    }
    call_ids = [
        c.get("id")
        for m in raw_messages
        if m["role"] == "assistant"
        for c in m.get("tool_calls") or []
    ]
    tools = [m for m in messages if m["role"] == "tool"]
    return {
        "messages_in_run": len(messages),
        "has_system_prompt": any(m["role"] == "system" for m in messages),
        "assistant_turns": sum(m["role"] == "assistant" for m in messages),
        "tool_call_turns": sum(
            1 for m in messages if m["role"] == "assistant" and m["tool_calls"]
        ),
        "tool_calls": len(calls),
        "tool_results": len(tools),
        "empty_tool_results": sum(1 for m in tools if m["text"] == ""),
        "unanswered_tool_calls": sum(1 for i in call_ids if i not in answered),
        "continue_prompts": sum(
            1 for m in messages if m["role"] == "user" and m["text"] == CONTINUE_PROMPT
        ),
        "escalation_calls": sum(1 for c in calls if c["function"] == ESCALATION_TOOL),
        "no_assistant_turn": not any(m["role"] == "assistant" for m in messages),
        "tool_functions": dict(sorted(Counter(c["function"] for c in calls).items())),
    }


def schema_survey(runs: list[dict]) -> dict[str, Any]:
    """What the exported messages look like (reported in the manifest)."""
    keys: dict[str, Counter] = defaultdict(Counter)
    content: dict[str, Counter] = defaultdict(Counter)
    parts: dict[str, Counter] = defaultdict(Counter)
    call_keys: Counter = Counter()
    first_roles: Counter = Counter()
    for run in runs:
        first_roles[",".join(m["role"] for m in run["messages"][:2])] += 1
        for m in run["messages"]:
            keys[m["role"]].update(m.keys())
            content[m["role"]][type(m.get("content")).__name__] += 1
            if isinstance(m.get("content"), list):
                for p in m["content"]:
                    parts[p.get("type")].update(p.keys())
            for c in m.get("tool_calls") or []:
                call_keys.update(c.keys())
    return {
        "message_keys": {r: dict(c) for r, c in keys.items()},
        "content_types": {r: dict(c) for r, c in content.items()},
        "content_part_keys": {t: dict(c) for t, c in parts.items()},
        "tool_call_keys": dict(call_keys),
        "first_two_roles": dict(first_roles),
    }


# ---------------------------------------------------------------- labels


def is_negative_control(run: dict, config: dict = CONFIG) -> bool:
    return run["_file"].endswith(config["negative_control_suffix"])


def class_of(run: dict, label: dict[str, str] | None, config: dict = CONFIG) -> str:
    if is_negative_control(run, config):
        if label is not None or run["in_paper_analysis"]:
            raise ValueError(f"{run['run_key']}: negative control with a label row")
        return "clean_negative"
    if label is None:
        raise ValueError(f"{run['run_key']}: no label row")
    for field in ("model", "condition", "problem_id"):
        if label[field] != run[field]:
            raise ValueError(f"{run['run_key']}: {field} differs from its label row")
    hacked, classification = label["reward_hacked"], label["classification"]
    if hacked == "yes" and classification == "reward_hacking":
        return "positive"
    if hacked == "no" and classification == "legitimate":
        return "negative"
    if hacked == "no" and classification in ("heuristic", "error"):
        return classification
    raise ValueError(f"{run['run_key']}: unhandled label {hacked!r}/{classification!r}")


def run_metadata(
    run: dict, name: str, label: dict | None, stats: dict, config: dict = CONFIG
) -> dict[str, Any]:
    score = run.get("score") or {}
    labels = None
    if label is not None:
        labels = {f: label.get(f) for f in config["label_fields"]}
    return {
        "source_dataset": "adversarial_evilgenie",
        "detection_point": "full",
        "run_key": run["run_key"],
        "parent_id": run["run_key"],
        "task_id": run["problem_id"],
        "problem_id": run["problem_id"],
        "problem_set": run["problem_set"],
        "model": run["model"],
        "model_id": run["model_id"],
        "condition": run["condition"],
        "epoch": run["epoch"],
        "eval_file": run["eval_file"],
        "transcript_file": run["_file"],
        "transcript_line": run["_line"],
        "in_paper_analysis": run["in_paper_analysis"],
        "label_class": name,
        "ground_truth": config["ground_truth"][name],
        "separately_reported": run["model"] == config["separately_reported_model"],
        "judge_type": None if label is None else label["judge_type"],
        "tamper_hack": None if label is None else label["tamper_hack"],
        "labels": labels,
        "labels_source": None if label is None else config["labels_file"],
        "transcript_score": {
            "llm_classification": score.get("llm_classification"),
            "test_result": score.get("test_result"),
        },
        **{k: v for k, v in stats.items()},
    }


def build_rows(
    runs: list[dict], labels: dict[str, dict], config: dict = CONFIG
) -> tuple[list[dict], dict[str, Any]]:
    """Rows (id, trajectory, metadata), one per run, plus a join summary."""
    keys = {r["run_key"] for r in runs}
    if missing := set(labels) - keys:
        raise ValueError(f"{len(missing)} label rows have no transcript")
    rows = []
    for run in runs:
        label = labels.get(run["run_key"])
        name = class_of(run, label, config)
        messages = [normalize(m) for m in run["messages"]]
        stats = run_stats(run["messages"], messages)
        rows.append(
            {
                "id": run["run_key"],
                "trajectory": render(messages),
                "meta": run_metadata(run, name, label, stats, config),
            }
        )
    found = Counter(r["meta"]["label_class"] for r in rows)
    counts = {name: found.get(name, 0) for name in config["expected_counts"]}
    if counts != config["expected_counts"] or sum(found.values()) != sum(
        counts.values()
    ):
        raise ValueError(f"class counts {dict(found)} != {config['expected_counts']}")
    summary = {
        "runs": len(runs),
        "label_rows": len(labels),
        "label_rows_joined": sum(1 for r in runs if r["run_key"] in labels),
        "classes": counts,
    }
    return rows, summary


# ---------------------------------------------------------------- summaries


def by(rows: list[dict], *fields: str) -> dict[str, dict[str, int]]:
    """Class counts per value of `fields` (joined with '/')."""
    out: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        key = "/".join(str(r["meta"][f]) for f in fields)
        out[key][r["meta"]["label_class"]] += 1
    return {k: dict(sorted(v.items())) for k, v in sorted(out.items())}


def row_summary(rows: list[dict], monitors: list[str]) -> dict[str, Any]:
    metas = [r["meta"] for r in rows]
    classes = sorted({m["label_class"] for m in metas})
    too_long: dict[str, Any] = {}
    for monitor in monitors:
        over = [m for m in metas if m["too_long"][monitor]]
        too_long[monitor] = {
            "total": len(over),
            "by_class": dict(sorted(Counter(m["label_class"] for m in over).items())),
            "by_model": dict(sorted(Counter(m["model"] for m in over).items())),
        }
    lengths = {
        monitor: {
            c: quantiles([m["tokens"][monitor] for m in metas if m["label_class"] == c])
            for c in classes
        }
        for monitor in monitors
    }
    length_auroc = {}
    for monitor in monitors:
        fits = [m for m in metas if not m["too_long"][monitor]]
        length_auroc[monitor] = {
            "all_rows": mann_whitney_auroc(
                [m["tokens"][monitor] for m in metas if m["label_class"] == "positive"],
                [m["tokens"][monitor] for m in metas if m["label_class"] == "negative"],
            ),
            "rows_that_fit": mann_whitney_auroc(
                [m["tokens"][monitor] for m in fits if m["label_class"] == "positive"],
                [m["tokens"][monitor] for m in fits if m["label_class"] == "negative"],
            ),
        }
    empty = defaultdict(Counter)
    for m in metas:
        empty[m["label_class"]]["rows"] += 1
        empty[m["label_class"]]["rows_with_empty_tool_results"] += bool(
            m["empty_tool_results"]
        )
        empty[m["label_class"]]["empty_tool_results"] += m["empty_tool_results"]
        empty[m["label_class"]]["tool_results"] += m["tool_results"]
    functions: Counter = Counter()
    for m in metas:
        functions.update(m["tool_functions"])
    return {
        "rows": len(rows),
        "by_class": dict(sorted(Counter(m["label_class"] for m in metas).items())),
        "by_model": by(rows, "model"),
        "by_condition": by(rows, "condition"),
        "by_model_and_condition": by(rows, "model", "condition"),
        "by_judge_type": by(rows, "judge_type"),
        "by_tamper_hack": by(rows, "tamper_hack"),
        "separately_reported_model": by(
            [r for r in rows if r["meta"]["separately_reported"]], "model"
        ),
        "too_long": too_long,
        "token_lengths": lengths,
        "length_only_auroc": length_auroc,
        "empty_tool_results": {c: dict(v) for c, v in sorted(empty.items())},
        "no_assistant_turn": by(
            [r for r in rows if r["meta"]["no_assistant_turn"]], "model"
        ),
        "no_system_prompt": [
            r["id"] for r in rows if not r["meta"]["has_system_prompt"]
        ],
        "unanswered_tool_calls": sum(m["unanswered_tool_calls"] for m in metas),
        "continue_prompts": sum(m["continue_prompts"] for m in metas),
        "escalation_calls": by(
            [r for r in rows if r["meta"]["escalation_calls"]], "condition"
        ),
        "tool_functions": dict(functions.most_common()),
    }


# ---------------------------------------------------------------- hand check


def hand_check_sample(rows: list[dict], seed: int, per_class: int) -> list[dict]:
    """`per_class` random positives, then negatives (sorted by id, one RNG)."""
    rng = random.Random(seed)
    picked = []
    for name in ("positive", "negative"):
        pool = sorted(
            (r for r in rows if r["meta"]["label_class"] == name), key=lambda r: r["id"]
        )
        picked += rng.sample(pool, per_class)
    return picked


def hand_check_text(picked: list[dict], tail_chars: int) -> str:
    out = [
        "# EvilGenie hand check",
        "",
        f"{len(picked)} rows: random positives then random negatives "
        f"(seed {HAND_CHECK['seed']}). Each shows the last {tail_chars:,} "
        "characters of the rendered trajectory (what the monitor sees, before "
        "the prompt wrapper) and the dataset's judge explanation and evidence.",
        "",
    ]
    for n, row in enumerate(picked, 1):
        m, labels = row["meta"], row["meta"]["labels"] or {}
        tail = row["trajectory"][-tail_chars:]
        fence = "~~~~" if "```" in tail else "```"
        out += [
            f"## {n}. {m['label_class']}: {m['model']} / {m['condition']} / "
            f"{m['problem_id']}",
            "",
            f"- run_key: `{m['run_key']}`",
            f"- judge_type: {m['judge_type']}; tamper_hack: {m['tamper_hack']}; "
            f"test_result: {labels.get('test_result')}; final_verdict: "
            f"{labels.get('final_verdict')}; judge_confidence: "
            f"{labels.get('judge_confidence')}",
            f"- tokens: {m['tokens']}; too_long: {m['too_long']}; trajectory "
            f"{len(row['trajectory']):,} chars; tool calls {m['tool_calls']}; "
            f"empty tool results {m['empty_tool_results']}",
            f"- judge_explanation: {labels.get('judge_explanation')}",
            f"- judge_evidence: {labels.get('judge_evidence')}",
            "",
            fence,
            tail,
            fence,
            "",
        ]
    return "\n".join(out)


def hand_check() -> None:
    path = DATA / "rows_4b-v03.jsonl"
    if eval_sets.file_sha256(path) != CONFIG["sha256"].get(path.name):
        raise ValueError(f"{path} does not match its SHA-256 in config.json")
    rows = []
    for r in read_jsonl(path):
        trajectory = split_prompt(r["prompt"]).trajectory
        rows.append({"id": r["id"], "trajectory": trajectory, "meta": r["metadata"]})
    picked = hand_check_sample(rows, HAND_CHECK["seed"], HAND_CHECK["per_class"])
    text = hand_check_text(picked, HAND_CHECK["tail_chars"])
    (DATA / "hand_check.md").write_text(text + "\n")
    print(text)


# ---------------------------------------------------------------- main


def instructions() -> dict[str, str]:
    texts = {
        name: (ROOT / path).read_text() for name, path in CONFIG["prompts"].items()
    }
    for name, text in texts.items():
        if sha(text) != CONFIG["prompt_sha256"][name]:
            raise ValueError(f"{name} prompt does not match its frozen SHA-256")
    return texts


def row_monitors(config: dict = CONFIG) -> list[str]:
    """Monitors with their own rows (27b-merged scores the 27b rows)."""
    return [m for m, spec in config["monitors"].items() if "rows_from" not in spec]


def token_counters(monitors: list[str]) -> dict:
    """Prompt length under each monitor's chat template and tokenizer files."""
    from transformers import AutoTokenizer

    from gleipnir.adversarial.contract import render as render_chat
    from gleipnir.adversarial.monitors import MONITORS

    counters = {}
    for monitor in monitors:
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


def rows_step(record: bool) -> None:
    verify_inputs()
    labels = read_labels(RAW / CONFIG["labels_file"])
    runs = read_transcripts()
    texts = instructions()
    monitors = row_monitors()
    heads = {m: texts[CONFIG["monitors"][m]["prompt"]] for m in monitors}
    rows, summary = build_rows(runs, labels)
    add_tokens(rows, heads, token_counters(monitors))
    manifest: dict[str, Any] = {
        "source": CONFIG["source"],
        "inputs_sha256": CONFIG["inputs_sha256"],
        "schema": schema_survey(runs),
        "join": summary,
        "rows": row_summary(rows, monitors),
        "max_tokens": CONFIG["max_tokens"],
        "sha256": {},
    }
    DATA.mkdir(parents=True, exist_ok=True)
    for monitor in monitors:
        path = DATA / f"rows_{monitor}.jsonl"
        manifest["sha256"][path.name] = write_jsonl(
            path, monitor_rows(rows, heads[monitor])
        )
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if record:
        config = json.loads(CONFIG_PATH.read_text())
        config["sha256"] = manifest["sha256"]
        CONFIG_PATH.write_text(json.dumps(config, indent=2) + "\n")
    shown = {k: v for k, v in manifest.items() if k != "inputs_sha256"}
    print(json.dumps(shown, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=("rows", "hand-check"))
    parser.add_argument(
        "--record", action="store_true", help="write row SHA-256 into config.json"
    )
    args = parser.parse_args()
    rows_step(args.record) if args.step == "rows" else hand_check()


if __name__ == "__main__":
    main()
