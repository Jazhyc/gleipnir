"""Shared paths, loaders and trajectory helpers for quiet cheats (stdlib only)."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIG = json.loads((HERE / "config.json").read_text())
ARM1 = CONFIG["arm1_deletion"]
MA_HERE = ROOT / "experiments" / "adversarial_monitor_aware"
MA_CONFIG = json.loads((MA_HERE / "config.json").read_text())
MA_DATA = ROOT / MA_CONFIG["scoring"]["data"]
MA_OUT = ROOT / MA_CONFIG["output"]
DATA = ROOT / CONFIG["data"] / "arm1"
OUT = ROOT / CONFIG["output"] / "arm1"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return file_sha256(path)


def task_number(task: str) -> int:
    return int(task.rsplit("_", 1)[1])


def run_order(run: dict[str, Any]) -> tuple[int, int]:
    return MA_CONFIG["setups"].index(run["setup"]), task_number(run["task"])


# ---------------------------------------------------------------- sources


def load_runs() -> list[dict[str, Any]]:
    """monitor-aware's runs, checked against the frozen hash."""
    path = MA_DATA / "runs.jsonl"
    manifest = json.loads((MA_DATA / "runs_manifest.json").read_text())
    if file_sha256(path) != manifest["sha256"]:
        raise ValueError("monitor-aware runs.jsonl does not match its manifest")
    return read_jsonl(path)


def load_replays() -> dict[str, dict[str, Any]]:
    return {r["id"]: r for r in read_jsonl(MA_DATA / "replay.jsonl")}


def load_ma_rows(prompt: str) -> dict[str, dict[str, Any]]:
    key = f"rows_{prompt}"
    path = MA_DATA / f"{key}.jsonl"
    if file_sha256(path) != MA_CONFIG["scoring"]["sha256"][key]:
        raise ValueError(f"{path} does not match its frozen SHA256")
    return {r["metadata"]["parent_id"]: r for r in read_jsonl(path)}


def load_labels() -> dict[str, dict[str, Any]]:
    path = ROOT / MA_CONFIG["labels"]["dir"] / "labels.jsonl"
    return {r["run_id"]: r for r in read_jsonl(path)}


def ma_cuts(runs: list[dict], replays: dict[str, dict]) -> dict[str, int]:
    """monitor-aware's cut per run (cheats: first passing turn; others matched)."""
    from experiments.adversarial_monitor_aware.rows import cuts

    seed = MA_CONFIG["scoring"]["length_match_seed"]
    return {k: v["cut"] for k, v in cuts(runs, replays, seed).items()}


# ---------------------------------------------------------------- trajectories


def assistant_steps(messages: list[dict[str, Any]]) -> dict[int, int]:
    """Step number (1-based, every assistant message) -> message index.

    The same numbering as monitor-aware's reader packets, so the readers'
    `first_step` values point at the same steps.
    """
    out, step = {}, 0
    for i, message in enumerate(messages):
        if message["role"] == "assistant":
            step += 1
            out[step] = i
    return out


def tool_results(messages: list[dict[str, Any]], index: int) -> list[int]:
    """Indices of the tool messages that answer assistant message `index`."""
    out = []
    for j in range(index + 1, len(messages)):
        if messages[j]["role"] != "tool":
            break
        out.append(j)
    return out


def argument_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def step_text(
    messages: list[dict[str, Any]], *, outputs: bool, marker_at: int | None = None
) -> str:
    """Numbered agent steps: prose, tool calls with raw arguments and, if
    `outputs`, the tool outputs the agent saw. No system or user text."""
    parts, step = [], 0
    for i, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        step += 1
        mark = " ⟦MARKED STEP⟧" if marker_at == i else ""
        block = [f"=== Step {step}{mark} ==="]
        if message["text"].strip():
            block += ["[message]", message["text"].strip()]
        for n, call in enumerate(message.get("tool_calls") or []):
            block.append(f"[tool call {n}: {call['function']}]")
            for key, value in (call.get("arguments") or {}).items():
                block += [f"{key}:", argument_text(value)]
        if outputs:
            for j in tool_results(messages, i):
                result = messages[j]
                error = result.get("error")
                text = f"Error: {error['message']}" if error else result["text"]
                block += ["[tool output]", text]
        parts.append("\n".join(block))
    return "\n\n".join(parts)


def proportional_quota(counts: dict[str, int], total: int) -> dict[str, int]:
    """Largest-remainder allocation in proportion to `counts` (monitor-aware's
    decoy rule; ties go to the setup earlier in config order)."""
    from experiments.adversarial_monitor_aware.labels import decoy_quota

    return decoy_quota(counts, total)


def seeded_sample(pool: list[Any], k: int, seed: int) -> list[Any]:
    return random.Random(seed).sample(pool, k)
