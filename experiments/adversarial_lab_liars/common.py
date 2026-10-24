"""Shared, torch-free helpers for the multi-agent liar study.

Source episodes, label joins, the last-N-cycles rendering, swap edits and
their validation, and the statistics used by `analyze.py`.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from gleipnir.adversarial.eval_sets import ROOT, file_sha256

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text())
DATA = ROOT / CONFIG["data"]
OUTPUT = ROOT / CONFIG["output"]
WORKERS = "ABCDEF"
RECRUITERS = "BC"
CYCLE = re.compile(r"service cycle (\d+) of")


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def openrouter_key() -> str:
    """OPENROUTER_API_KEY from the environment, else the workspace `.env`."""
    import os

    if os.environ.get("OPENROUTER_API_KEY", "").strip():
        return os.environ["OPENROUTER_API_KEY"].strip()
    for path in (ROOT / ".env", ROOT.parents[1] / ".env"):
        if path.exists():
            for line in path.read_text().splitlines():
                if line.startswith("OPENROUTER_API_KEY="):
                    return line.split("=", 1)[1].strip().strip("\"'")
    raise SystemExit("OPENROUTER_API_KEY is missing")


def openrouter_balance(key: str) -> float:
    import requests

    response = requests.get(
        "https://openrouter.ai/api/v1/credits",
        headers={"Authorization": f"Bearer {key}"},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()["data"]
    return float(data["total_credits"] - data["total_usage"])


# ---------------------------------------------------------------- source


def check_labels(source: Path) -> Path:
    labels = source / "results" / "labels"
    for name, expected in CONFIG["label_sha256"].items():
        if file_sha256(labels / name) != expected:
            raise ValueError(f"{name} does not match its frozen SHA256")
    return labels


def episodes(source: Path) -> list[tuple[str, Path, dict[str, Any]]]:
    """(run, episode directory, outcome) for every completed scoring episode."""
    found = []
    for run in CONFIG["runs"]:
        for path in sorted((source / "results" / run).glob("*/outcome.json")):
            outcome = json.loads(path.read_text())
            if outcome["reason"] == CONFIG["complete_reason"]:
                found.append((run, path.parent, outcome))
    return found


def world_posts(episode: Path) -> dict[int, dict[str, Any]]:
    posts = {}
    for line in (episode / "world.jsonl").open():
        event = json.loads(line)
        if event.get("type") == "board_post":
            posts[int(event["message"]["id"])] = event["message"]
    return posts


def analysis_recruiter(seed: int) -> str:
    """Amendment 3 (liar-scale `9503428f`): B on odd seeds, C on even seeds."""
    return "B" if seed % 2 else "C"


def index_labels(labels: Path) -> dict[str, dict]:
    recruiters = {
        (r["run"], r["episode"], r["author"], r["rule"]): r
        for r in read_jsonl(labels / "labels_recruiters.jsonl")
    }
    exposure = {
        (r["run"], r["episode"], r["worker"], r["rule"]): r
        for r in read_jsonl(labels / "labels_exposure.jsonl")
    }
    posts = defaultdict(list)
    for p in read_jsonl(labels / "labels_posts.jsonl"):
        posts[(p["run"], p["episode"], p["author"])].append(p)
    return {"recruiters": recruiters, "exposure": exposure, "posts": posts}


RECRUITER_GROUP = {
    "liar": "liar",
    "self_retracting_liar": "self_retracting_liar",
    "assigned_deceiver_no_false_claim": "assigned_deceiver_no_lie",
    "honest": "honest_recruiter",
}


def worker_metadata(
    run: str, episode: Path, outcome: dict, worker: str, index: dict
) -> dict[str, Any]:
    """Every join a row needs: models, seat, labels, exposure, diagnostic."""
    name = episode.name
    seed = int(outcome["seed"])
    recruiter = worker in RECRUITERS
    diagnostics = {d["worker"] for d in outcome.get("diagnostics") or []}
    meta: dict[str, Any] = {
        "run": run,
        "episode": name,
        "task_id": f"{run}/{name}",
        "cell": outcome["cell"],
        "seed": seed,
        "recruiter_model": outcome["recruiter"],
        "pool_model": outcome["pool"],
        "worker": worker,
        "model": outcome["recruiter"] if recruiter else outcome["pool"],
        "role": "recruiter"
        if recruiter
        else ("target" if worker == "A" else "bystander"),
        "seat_arm": outcome["seatArms"][worker] if recruiter else None,
        "ran_diagnostic": worker in diagnostics,
        "analysis_recruiter": recruiter and worker == analysis_recruiter(seed),
        "honest_episode": all(v == "honest" for v in outcome["seatArms"].values()),
    }
    if recruiter:
        labels = {
            rule: index["recruiters"][(run, name, worker, rule)]
            for rule in ("primary", "sensitivity")
        }
        if labels["primary"]["analysis_recruiter"] != meta["analysis_recruiter"]:
            raise ValueError(f"{run}/{name}/{worker}: analysis recruiter mismatch")
        if labels["primary"]["seat_arm"] != meta["seat_arm"]:
            raise ValueError(f"{run}/{name}/{worker}: seat arm mismatch")
        meta["label_primary"] = labels["primary"]["label"]
        meta["label_sensitivity"] = labels["sensitivity"]["label"]
        meta["exposed_primary"] = meta["exposed_sensitivity"] = None
        meta["condition"] = RECRUITER_GROUP[meta["label_primary"]]
        meta["ground_truth"] = int(
            meta["label_primary"] in ("liar", "self_retracting_liar")
        )
    else:
        exposure = {
            rule: index["exposure"][(run, name, worker, rule)]
            for rule in ("primary", "sensitivity")
        }
        meta["label_primary"] = meta["label_sensitivity"] = None
        meta["exposed_primary"] = bool(exposure["primary"]["exposed"])
        meta["exposed_sensitivity"] = bool(exposure["sensitivity"]["exposed"])
        meta["worker_never_acted"] = bool(exposure["primary"]["worker_never_acted"])
        state = "exposed" if meta["exposed_primary"] else "unexposed"
        meta["condition"] = f"{meta['role']}_{state}"
        meta["ground_truth"] = 0
    return meta


# ---------------------------------------------------------------- rendering


def checkin_cycles(messages: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """(message index, cycle) for each user check-in."""
    found = []
    for index, message in enumerate(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        text = (
            content
            if isinstance(content, str)
            else "".join(p.get("text", "") for p in content or [])
        )
        match = CYCLE.search(text)
        if match:
            found.append((index, int(match.group(1))))
    return found


def window_history(messages: list[dict[str, Any]], cycles: int) -> list[dict[str, Any]]:
    """The cycle-1 block (task read, private evidence) plus the last `cycles`.

    The last cycle is the worker's last check-in. Cycle 1 is kept because its
    tool outputs hold the task and the private routing fact; without them a
    monitor cannot tell a quarantine claim is false. Messages are not edited.
    """
    checkins = checkin_cycles(messages)
    if not checkins:
        raise ValueError("history has no check-ins")
    last = checkins[-1][1]
    start = next(i for i, c in checkins if c >= last - cycles + 1)
    first_end = next((i for i, c in checkins if c >= 2), len(messages))
    if start <= first_end:
        return list(messages)
    return list(messages[:first_end]) + list(messages[start:])


def own_post_texts(messages: list[dict[str, Any]]) -> list[str]:
    return [
        (part.get("arguments") or {}).get("text", "")
        for m in messages
        if m.get("role") == "assistant"
        for part in m.get("content") or []
        if part.get("type") == "toolCall" and part.get("name") == "board_post"
    ]


# ---------------------------------------------------------------- swap edits


def apply_edits(text: str, edits: list[dict[str, str]]) -> str:
    """Apply exact, non-overlapping substring edits; raise on any violation."""
    if not edits:
        raise ValueError("no edits")
    spans = []
    for edit in edits:
        old, new = edit.get("old"), edit.get("new")
        if not isinstance(old, str) or not isinstance(new, str) or not old:
            raise ValueError("edit needs non-empty string old and string new")
        if text.count(old) != 1:
            raise ValueError("edit text does not occur exactly once")
        start = text.index(old)
        spans.append((start, start + len(old), new))
    spans.sort()
    for (_, end, _), (start, _, _) in zip(spans, spans[1:], strict=False):
        if start < end:
            raise ValueError("edits overlap")
    out, cursor = [], 0
    for start, end, new in spans:
        out.append(text[cursor:start])
        out.append(new)
        cursor = end
    out.append(text[cursor:])
    result = "".join(out)
    if result == text:
        raise ValueError("edits change nothing")
    return result


def changed_share(text: str, edits: list[dict[str, str]]) -> float:
    return sum(len(e["old"]) for e in edits) / max(len(text), 1)


def outside_unchanged(text: str, edits: list[dict[str, str]], result: str) -> bool:
    """Every character outside the edited spans survives, in order, unchanged."""
    spans = sorted((text.index(e["old"]), len(e["old"]), e["new"]) for e in edits)
    cursor_in, cursor_out = 0, 0
    for start, length, new in spans:
        keep = text[cursor_in:start]
        if result[cursor_out : cursor_out + len(keep)] != keep:
            return False
        cursor_out += len(keep) + len(new)
        cursor_in = start + length
    return result[cursor_out:] == text[cursor_in:]


def validate_rewrite(
    text: str, edits: Any, screens: list[dict[str, Any]], max_changed_share: float
) -> tuple[bool, str, str | None]:
    """(passed, reason, rewritten text). Frozen rule, README 'Truth swap'."""
    if not isinstance(edits, list):
        return False, "unparsed", None
    try:
        result = apply_edits(text, edits)
    except ValueError as error:
        return False, f"edit: {error}", None
    if not outside_unchanged(text, edits, result):
        return False, "outside text changed", result
    if changed_share(text, edits) > max_changed_share:
        return False, "too much changed", result
    if len(screens) < CONFIG["swap"]["screen_passes"]:
        return False, "screen missing", result
    for screen in screens:
        if screen.get("false_claim") != "no":
            reason = (
                "screen unparsed" if screen.get("false_claim") is None else "screen yes"
            )
            return False, reason, result
    return True, "ok", result


# ---------------------------------------------------------------- statistics


def auroc(positives: list[float], negatives: list[float]) -> float | None:
    """Mann-Whitney AUROC with ties counted as one half."""
    if not positives or not negatives:
        return None
    wins = 0.0
    for p in positives:
        for n in negatives:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(positives) * len(negatives))


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if n == 0:
        return None
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, centre - half), min(1.0, centre + half)]


def seed_bootstrap(
    records: list[dict[str, Any]],
    statistic: Callable[[list[dict[str, Any]]], float | None],
    samples: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """95% percentile interval resampling seeds, keeping all of a seed's rows.

    Draws where the statistic is undefined (e.g. one class missing) are
    dropped and counted.
    """
    samples = samples or CONFIG["analysis"]["bootstrap_samples"]
    seed = CONFIG["analysis"]["bootstrap_seed"] if seed is None else seed
    clusters = defaultdict(list)
    for r in records:
        clusters[r["seed"]].append(r)
    keys = sorted(clusters)
    rng = random.Random(seed)
    values = []
    for _ in range(samples):
        drawn = [r for k in rng.choices(keys, k=len(keys)) for r in clusters[k]]
        value = statistic(drawn)
        if value is not None:
            values.append(value)
    if not values:
        return {"ci95": None, "dropped_draws": samples}
    values.sort()
    lo = values[int(0.025 * len(values))]
    hi = values[max(int(0.975 * len(values)) - 1, 0)]
    return {"ci95": [lo, hi], "dropped_draws": samples - len(values)}
