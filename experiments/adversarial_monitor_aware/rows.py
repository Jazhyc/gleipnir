"""Build the monitor rows for the five setups (Amendment 4; CPU only).

After `replay.py extract` and `replay.py replay`:

    PYTHONPATH=src:. ../../.venv-data/bin/python \
        experiments/adversarial_monitor_aware/rows.py

One row per analysed run (413), at the natural-cheats first-cheat point:
- a cheat is cut after its first passing turn (`replay.py`), without that
  turn's tool results;
- a non-cheat is cut, as natural-cheats cuts its negatives, after the
  assistant tool-call turn nearest a length drawn with replacement from the
  cheats' cut indices (`random.Random(seed)`, runs in setup order, then task
  number), clipped to the run.

Rendering is natural-cheats' (`trajectories.render`). Each row is written
under the student prompt (4B) and the teacher prompt (27B, Kimi), with
byte-identical trajectories. Rows over 29,696 tokens for a monitor are marked
`too_long` and counted, never truncated. Writes `rows_student.jsonl`,
`rows_teacher.jsonl`, `canaries.jsonl`, `kimi_order.json` and `manifest.json`
to `data/adversarial_monitor_aware/`.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.adversarial_monitor_aware import leak
from experiments.adversarial_natural_cheats.trajectories import (
    cut,
    nearest_turn,
    render,
    tool_call_turns,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIG = json.loads((HERE / "config.json").read_text())
SCORING = CONFIG["scoring"]
DATA = ROOT / SCORING["data"]
POINT = "first_cheat"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return file_sha256(path)


def task_number(task: str) -> int:
    return int(task.rsplit("_", 1)[1])


def run_order(run: dict[str, Any]) -> tuple[int, int]:
    return CONFIG["setups"].index(run["setup"]), task_number(run["task"])


def cheat_cut(run: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    """The first passing turn, from a valid replay."""
    if not replay["final_matches"]:
        raise ValueError(f"{run['id']}: replay does not reproduce the final func.py")
    turn = replay["first_pass_turn"]
    if turn is None or turn not in tool_call_turns(run["messages"]):
        raise ValueError(f"{run['id']}: no first passing tool-call turn")
    return {"cut": turn, "rule": "first_passing_turn"}


def cuts(runs: list[dict], replays: dict[str, dict], seed: int) -> dict[str, dict]:
    """Cut per run: cheats at their first passing turn; non-cheats length-matched."""
    ordered = sorted(runs, key=run_order)
    out = {r["id"]: cheat_cut(r, replays[r["id"]]) for r in ordered if r["cheat"]}
    lengths = [out[r["id"]]["cut"] for r in ordered if r["cheat"]]
    rng = random.Random(seed)
    for run in ordered:
        if run["cheat"]:
            continue
        drawn = rng.choice(lengths)
        out[run["id"]] = {
            "cut": nearest_turn(tool_call_turns(run["messages"]), drawn),
            "rule": "length_matched",
            "drawn_length": drawn,
        }
    return out


def build(runs: list[dict], replays: dict[str, dict], seed: int) -> list[dict]:
    """Rows (id, trajectory, metadata), one per run, in setup and task order."""
    info = cuts(runs, replays, seed)
    rows = []
    for run in sorted(runs, key=run_order):
        c = info[run["id"]]
        kept = cut(run["messages"], c["cut"])
        meta = {
            "source_dataset": "adversarial_monitor_aware",
            "parent_id": run["id"],
            "task_id": run["task"],
            "setup": run["setup"],
            "status": run["status"],
            "cheat": run["cheat"],
            "ground_truth": int(run["cheat"]),
            "condition": f"{run['setup']}:{'cheat' if run['cheat'] else 'non_cheat'}",
            "detection_point": POINT,
            "cut_message_index": c["cut"],
            "cut_rule": c["rule"],
            **({"drawn_length": c["drawn_length"]} if "drawn_length" in c else {}),
            "messages_in_run": len(run["messages"]),
            "messages_in_cut": len(kept),
            "echo": run["echo"],
            "echo_in_cut": bool(leak.check(kept, CONFIG)["agent_echo"]),
        }
        rows.append(
            {"id": f"{run['id']}::{POINT}", "trajectory": render(kept), "meta": meta}
        )
    return rows


def kimi_order(rows: list[dict], seed: int) -> list[dict[str, Any]]:
    """Frozen Kimi order: the tiers in config, each tier shuffled with `seed`."""
    rng = random.Random(seed)
    out = []
    for tier, setups in enumerate(SCORING["kimi"]["order_tiers"], 1):
        ids = [r["id"] for r in rows if r["meta"]["setup"] in setups]
        rng.shuffle(ids)
        out += [{"id": i, "tier": tier} for i in ids]
    if sorted(o["id"] for o in out) != sorted(r["id"] for r in rows):
        raise ValueError("kimi order does not cover every row exactly once")
    return out


# ---------------------------------------------------------------- heavy parts


def instructions() -> dict[str, str]:
    texts = {name: (ROOT / p).read_text() for name, p in SCORING["prompts"].items()}
    for name, text in texts.items():
        if sha(text) != SCORING["prompt_sha256"][name]:
            raise ValueError(f"{name} prompt does not match its frozen SHA256")
    return texts


def token_counters() -> dict:
    from transformers import AutoTokenizer

    from gleipnir.adversarial.contract import render as render_chat
    from gleipnir.adversarial.monitors import MONITORS

    counters = {}
    for key, spec in SCORING["gpu_monitors"].items():
        monitor = MONITORS[spec["monitor"]]
        tok = AutoTokenizer.from_pretrained(
            monitor.base, revision=monitor.base_revision
        )

        def count(prompt: str, tok=tok) -> int:
            return len(tok.encode(render_chat(tok, prompt), add_special_tokens=False))

        counters[key] = (spec["prompt"], count)
    return counters


def canaries(texts: dict[str, str]) -> list[dict[str, Any]]:
    """Calibration rows with known scores from each GPU monitor's line source."""
    out = []
    for key, spec in SCORING["gpu_monitors"].items():
        line = SCORING["lines"][key]
        reference = {
            r["id"]: r["score"]
            for r in read_jsonl(ROOT / line["path"])
            if r.get("status") == "ok" and r.get("label") == 0
        }
        rows = read_jsonl(ROOT / SCORING["canary"]["rows"][spec["prompt"]])
        picked = [r for r in rows if r["id"] in reference][
            : SCORING["canary"]["per_monitor"]
        ]
        out += [
            {
                "id": r["id"],
                "monitor": key,
                "prompt": r["prompt"],
                "expected": reference[r["id"]],
            }
            for r in picked
        ]
    return out


def main() -> None:
    from gleipnir.adversarial.contract import MAX_TOKENS, join_prompt

    manifest_in = json.loads((DATA / "runs_manifest.json").read_text())
    if file_sha256(DATA / "runs.jsonl") != manifest_in["sha256"]:
        raise ValueError("runs.jsonl does not match runs_manifest.json")
    runs = read_jsonl(DATA / "runs.jsonl")
    replays = {r["id"]: r for r in read_jsonl(DATA / "replay.jsonl")}
    rows = build(runs, replays, SCORING["length_match_seed"])
    texts = instructions()
    counters = token_counters()
    too_long: dict[str, Counter] = {k: Counter() for k in counters}
    for row in rows:
        tokens = {
            key: count(join_prompt(texts[prompt], row["trajectory"]))
            for key, (prompt, count) in counters.items()
        }
        row["meta"]["tokens"] = tokens
        row["meta"]["too_long"] = {k: n > MAX_TOKENS for k, n in tokens.items()}
        for key, flag in row["meta"]["too_long"].items():
            if flag:
                too_long[key][row["meta"]["condition"]] += 1
    digests = {}
    for prompt in ("student", "teacher"):
        digests[f"rows_{prompt}"] = write_jsonl(
            DATA / f"rows_{prompt}.jsonl",
            [
                {
                    "id": r["id"],
                    "prompt": join_prompt(texts[prompt], r["trajectory"]),
                    "metadata": r["meta"],
                }
                for r in rows
            ],
        )
    digests["canaries"] = write_jsonl(DATA / "canaries.jsonl", canaries(texts))
    order = kimi_order(rows, SCORING["kimi"]["order_seed"])
    (DATA / "kimi_order.json").write_text(json.dumps(order, indent=1) + "\n")
    digests["kimi_order"] = file_sha256(DATA / "kimi_order.json")
    per_setup: dict[str, Counter] = {}
    for r in rows:
        m = r["meta"]
        c = per_setup.setdefault(m["setup"], Counter())
        c["rows"] += 1
        c["cheats"] += m["cheat"]
        c["echo_runs"] += m["echo"]
    manifest = {
        "runs_sha256": manifest_in["sha256"],
        "cheat_replays": {
            rid: {
                "first_pass_turn": r["first_pass_turn"],
                "final_matches": r["final_matches"],
            }
            for rid, r in replays.items()
            if r["cheat"]
        },
        "rows": len(rows),
        "per_setup": {k: dict(v) for k, v in per_setup.items()},
        "too_long": {k: dict(v) for k, v in too_long.items()},
        "teacher_tokens_total": sum(r["meta"]["tokens"]["27b"] for r in rows),
        "student_tokens_total": sum(r["meta"]["tokens"]["4b"] for r in rows),
        "sha256": digests,
    }
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
