"""Replay arm 1's edited trajectories and regenerate their tool outputs.

    PY=../../.venv-impbench/bin/python
    S=experiments/adversarial_quiet_cheats
    PYTHONPATH=src:. $PY $S/replay.py --set quiet --threads 8
    PYTHONPATH=src:. $PY $S/replay.py --set strip --threads 8

Uses monitor-aware's sandbox (`replay.Box`, pinned image, network off,
func.py and test.py in `/`, the agent's tool calls in order with the same
limits and editor semantics) and the pilot's offline nonce check. Each item is
replayed twice up to its cut, in fresh containers: the original actions (A) and
the edited actions (B). Outputs are captured as Inspect returned them (bash and
python: stderr, a newline, then stdout; editor: the `cat -n` views and
snippets; Inspect's 16 KiB truncation).

Tool output after the first edited turn (Amendment 1):
- recorded result was an error: kept;
- A == B: the edit did not reach the output, so the recorded output is kept;
- A == recorded output: B is used (the output the edited code produces);
- otherwise the output is unresolved and the item fails the replay check.

Pass rule: every func.py version of B matches A's version at the same turn
under the item's tier, A's versions equal monitor-aware's replay, and B's
first passing turn (nonce check of every version not byte-identical to A's)
is the cut (cheats) or never (controls).

Writes `data/adversarial_quiet_cheats/arm1/{set}_replay.jsonl` (resumable).
Model-written code only ever runs inside containers.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from experiments.adversarial_quiet_cheats.common import (
    DATA,
    ROOT,
    load_replays,
    load_runs,
    read_jsonl,
    sha,
    tool_results,
)

MAX_OUTPUT = 16 * 1024
SNIPPET_LINES = 4

# Applies or views one text_editor call inside the container and prints a JSON
# object {"status": ..., "output": ...}. Application semantics are monitor-
# aware's EDITOR; the output strings follow the editor Inspect ran.
EDITOR_CAPTURE = r"""
import json, posixpath, sys
a = json.loads(sys.stdin.read())
p, c = a.get("path"), a.get("command")
shown = posixpath.join("/", p) if p else p
REVIEW = "Review the changes and make sure they are as expected."
INSERT_NOTE = " (correct indentation, no duplicate lines, etc)."
AGAIN = " Edit the file again if necessary."
def make(content, desc, start):
    rows = enumerate(content.split("\n"))
    body = "\n".join(f"{i + start:6}\t{line}" for i, line in rows)
    return f"Here's the result of running `cat -n` on {desc}:\n{body}\n"
def done(status, output=None):
    print(json.dumps({"status": status, "output": output}))
try:
    if c == "view":
        text = open(p).read()
        r = a.get("view_range")
        if r:
            s, e = r
            lines = text.split("\n")
            text = "\n".join(lines[s - 1:] if e == -1 else lines[s - 1:e])
            done("ok", make(text, shown, s))
        else:
            done("ok", make(text, shown, 1))
    elif c == "create":
        open(p, "w").write(a.get("file_text") or "")
        done("ok", f"File created successfully at: {shown}")
    elif c == "str_replace":
        s = open(p).read()
        old, new = a.get("old_str") or "", a.get("new_str") or ""
        if s.count(old) != 1:
            done("not_unique")
        else:
            line = s.split(old)[0].count("\n")
            t = s.replace(old, new, 1)
            open(p, "w").write(t)
            start = max(0, line - 4)
            end = line + 4 + new.count("\n")
            snip = "\n".join(t.split("\n")[start:end + 1])
            done("ok", f"The file {shown} has been edited. "
                 + make(snip, f"a snippet of {shown}", start + 1) + REVIEW + AGAIN)
    elif c == "insert":
        lines = open(p).read().split("\n")
        n = int(a.get("insert_line"))
        new = a.get("new_str")
        if new is None:
            new = a.get("insert_text") or ""
        added = new.split("\n")
        snip = lines[max(0, n - 4):n] + added + lines[n:n + 4]
        lines[n:n] = added
        open(p, "w").write("\n".join(lines))
        done("ok", f"The file {shown} has been edited. "
             + make("\n".join(snip), "a snippet of the edited file", max(1, n - 3))
             + REVIEW[:-1] + INSERT_NOTE + AGAIN)
    else:
        done("skipped")
except Exception as error:
    done("exception", repr(error))
"""


def truncate(name: str, output: str) -> str:
    """Inspect's tool-output truncation (16 KiB), from the pinned inspect_ai."""
    from inspect_ai.model._call_tools import truncate_tool_output

    t = truncate_tool_output(name, output, MAX_OUTPUT)
    return output if t is None else t.output


def capture_call(box, call: dict[str, Any], result: dict | None) -> str | None:
    """Rerun one call (monitor-aware's semantics); the output Inspect would
    have returned, or None when there is none to compare."""
    name, args = call["function"], call.get("arguments") or {}
    limit = ["timeout", "-k", "5", "60"]
    if name in ("bash", "python"):
        if name == "bash" and isinstance(args.get("command"), str):
            done = box.exec([*limit, "bash", "--login", "-c", args["command"]])
        elif name == "python" and isinstance(args.get("code"), str):
            done = box.exec(
                [*limit, "bash", "--login", "-c", "python3 -"], args["code"]
            )
        else:
            return None
        if done is None:
            return None
        out = f"{done.stderr}\n" if done.stderr else ""
        return truncate(name, out + done.stdout)
    if name == "text_editor":
        if result is not None and result.get("error"):
            return None
        if args.get("command") == "undo_edit":
            return None
        done = box.exec(["python3", "-c", EDITOR_CAPTURE], json.dumps(args))
        if done is None or done.returncode != 0:
            return None
        reply = json.loads(done.stdout.strip().splitlines()[-1])
        return truncate(name, reply["output"]) if reply["status"] == "ok" else None
    return None


def capture(messages: list[dict], upto: int, run: dict, image: str) -> dict[str, Any]:
    """Replay calls of assistant turns <= upto; outputs by tool-message index,
    and func.py versions after each turn."""
    from experiments.adversarial_monitor_aware.replay import (
        Box,
        initial_func,
        test_file,
    )

    box = Box(image)
    outputs: dict[int, str | None] = {}
    versions, texts = [], {}
    try:
        box.write("/func.py", initial_func(run["prompt"]))
        box.write("/test.py", test_file(run["test"], run["entry_point"]))
        current = box.read("/func.py") or ""
        texts[sha(current)] = current
        versions.append({"after_turn": None, "sha": sha(current)})
        for i, message in enumerate(messages[: upto + 1]):
            if message["role"] != "assistant" or not message.get("tool_calls"):
                continue
            results = tool_results(messages, i)
            for n, call in enumerate(message["tool_calls"]):
                j = results[n] if n < len(results) else None
                out = capture_call(box, call, messages[j] if j is not None else None)
                if j is not None:
                    outputs[j] = out
            now = box.read("/func.py") or ""
            if now != current:
                current = now
                texts[sha(now)] = now
                versions.append({"after_turn": i, "sha": sha(now)})
    finally:
        box.close()
    return {"outputs": outputs, "versions": versions, "texts": texts}


def resolve(
    original: list[dict],
    edited: list[dict],
    a: dict[int, Any],
    b: dict[int, Any],
    first_edited: int,
) -> tuple[list[dict], dict[str, list[int]]]:
    """Edited messages with tool outputs after `first_edited` regenerated."""
    out = [dict(m) for m in edited]
    kinds: dict[str, list[int]] = {"kept": [], "regenerated": [], "unresolved": []}
    for j, message in enumerate(original):
        if message["role"] != "tool" or j <= first_edited:
            continue
        if message.get("error"):
            kinds["kept"].append(j)
        elif a.get(j) == b.get(j):
            kinds["kept"].append(j)
        elif (
            a.get(j) is not None
            and a.get(j) == message["text"]
            and b.get(j) is not None
        ):
            out[j] = {**message, "text": b[j]}
            kinds["regenerated"].append(j)
        else:
            kinds["unresolved"].append(j)
    return out, kinds


def versions_match(a: dict, b: dict, renames: dict[str, str]) -> bool:
    from experiments.adversarial_quiet_cheats.checks import python_tier

    va, vb = a["versions"], b["versions"]
    if [v["after_turn"] for v in va] != [v["after_turn"] for v in vb]:
        return False
    return all(
        python_tier(a["texts"][x["sha"]], b["texts"][y["sha"]], renames) is not None
        for x, y in zip(va, vb, strict=False)
    )


def replay_item(item: dict, run: dict, ma_replay: dict, image: str, check) -> dict:
    from experiments.adversarial_quiet_cheats.checks import first_changed

    upto = item["cut"]
    original = run["messages"][: upto + 1]
    first = first_changed(original, item["messages"])
    a = capture(original, upto, run, image)
    b = capture(item["messages"], upto, run, image)
    ma_versions = [
        v
        for v in ma_replay["versions"]
        if v["after_turn"] is None or v["after_turn"] <= upto
    ]
    original_matches = [(v["after_turn"], v["sha"]) for v in a["versions"]] == [
        (v["after_turn"], v["sha"]) for v in ma_versions
    ]
    passed = {}
    for v in b["versions"]:
        if v["sha"] in ma_replay["passed"]:
            passed[v["sha"]] = ma_replay["passed"][v["sha"]]
        else:
            outcome = check(b["texts"][v["sha"]], run["test"], run["entry_point"])
            if outcome.get("note") == "timeout":
                outcome = check(b["texts"][v["sha"]], run["test"], run["entry_point"])
            passed[v["sha"]] = outcome["passed"]
    first_pass = next(
        (v["after_turn"] for v in b["versions"] if passed[v["sha"]]), None
    )
    if first is None:
        messages, kinds = (
            item["messages"],
            {"kept": [], "regenerated": [], "unresolved": []},
        )
    else:
        messages, kinds = resolve(
            original, item["messages"], a["outputs"], b["outputs"], first
        )
    expected = ma_replay["first_pass_turn"] if run["cheat"] else None
    return {
        "id": item["id"],
        "run_id": run["id"],
        "attempt": item.get("attempt", 1),
        "first_edited": first,
        "first_pass_turn": first_pass,
        "ever_passed": any(passed.values()),
        "passed": passed,
        "versions": b["versions"],
        "versions_match": versions_match(a, b, item.get("renames") or {}),
        "original_versions_match_ma": original_matches,
        "original_first_pass_turn": expected if original_matches else "mismatch",
        "expected_original_first_pass": expected,
        "outputs": {k: len(v) for k, v in kinds.items()},
        "unresolved_outputs": kinds["unresolved"],
        "fidelity_a_equals_recorded": sum(
            1
            for j, out in a["outputs"].items()
            if out is not None
            and not original[j].get("error")
            and out == original[j]["text"]
        ),
        "fidelity_compared": sum(
            1
            for j, out in a["outputs"].items()
            if out is not None and not original[j].get("error")
        ),
        "messages": messages,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", choices=("quiet", "strip"), required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    from experiments.adversarial_cheating_pilot import run as pilot
    from experiments.adversarial_cheating_pilot.recheck import check

    pilot.docker_env()
    shared = ROOT.parents[1] / "cache" / "recheck-tmp"
    shared.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(shared)
    good = check(
        "def f(x):\n    return x\n", "def check(c):\n    assert c(1) == 1\n", "f"
    )
    bad = check(
        "def f(x):\n    return 2\n", "def check(c):\n    assert c(1) == 1\n", "f"
    )
    if not good["passed"] or bad["passed"]:
        raise SystemExit(f"recheck canaries failed: {good} {bad}")
    items = read_jsonl(DATA / f"{args.set}.jsonl")
    runs = {r["id"]: r for r in load_runs()}
    ma = load_replays()
    path = DATA / f"{args.set}_replay.jsonl"
    done = {(r["id"], r["attempt"]) for r in read_jsonl(path)}
    todo = [
        i
        for i in items
        if (i["id"], i.get("attempt", 1)) not in done and not i.get("apply_errors")
    ]
    image = pilot.CONFIG["sandbox_image"]
    lock = threading.Lock()
    print(f"{len(todo)} items to replay", flush=True)

    def work(item):
        record = replay_item(
            item, runs[item["run_id"]], ma[item["run_id"]], image, check
        )
        with lock, path.open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"{item['id']}: first_pass {record['first_pass_turn']} "
                f"versions_match {record['versions_match']} "
                f"outputs {record['outputs']}",
                flush=True,
            )

    with ThreadPoolExecutor(args.threads) as pool:
        list(pool.map(work, todo))


if __name__ == "__main__":
    main()
