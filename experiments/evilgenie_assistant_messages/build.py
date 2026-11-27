"""EvilGenie states that keep the agent's assistant messages.

``PYTHONPATH=src:. python -m experiments.evilgenie_assistant_messages.build``

Same runs, labels and rendering as ``experiments/evilgenie_api_baselines``
(tool calls, tool results and user turns), plus the text parts of each
assistant message, placed as in the training data: ``[ASSISTANT]``, the text,
a blank line, then ``[TOOL CALLS]``. Reasoning parts stay dropped. A run
without assistant text renders byte for byte as its baseline state.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.adversarial_natural_cheats.trajectories import call_text, render

ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()
)
CONFIG = json.loads(Path(__file__).with_name("config.json").read_text())


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r) + "\n" for r in rows)
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def render_with_text(messages: list[dict[str, Any]]) -> str:
    """``trajectories.render`` with the assistant text kept."""
    blocks = []
    for message in messages:
        role = message["role"]
        if role == "assistant":
            parts = []
            if text := message["text"].strip():
                parts.append(text)
            calls = [call_text(call) for call in message.get("tool_calls") or []]
            if calls:
                parts.append("[TOOL CALLS]\n" + "\n".join(calls))
            if parts:
                blocks.append("[ASSISTANT]\n" + "\n\n".join(parts))
        elif block := render([message]):
            blocks.append(block)
    return "\n\n".join(blocks)


def has_text(messages: list[dict[str, Any]]) -> bool:
    return any(m["role"] == "assistant" and m["text"].strip() for m in messages)


def main() -> None:
    from experiments.adversarial_evilgenie import build as eg

    baseline_path = ROOT / CONFIG["baseline_states"]
    if (
        hashlib.sha256(baseline_path.read_bytes()).hexdigest()
        != CONFIG["baseline_states_sha256"]
    ):
        raise ValueError("baseline states differ from their frozen SHA-256")
    baseline = {r["id"]: r for r in read_jsonl(baseline_path)}
    eg.verify_inputs(eg.RAW)
    runs = {r["run_key"]: r for r in eg.read_transcripts(eg.RAW)}
    states, changed = [], Counter()
    for row_id, base in baseline.items():
        messages = [eg.normalize(m) for m in runs[row_id]["messages"]]
        if render(messages) != base["state"]:
            raise ValueError(f"{row_id}: baseline rendering not reproduced")
        state = render_with_text(messages)
        with_text = has_text(messages)
        if not with_text and state != base["state"]:
            raise ValueError(f"{row_id}: changed without assistant text")
        changed[(base["metadata"]["label_class"], with_text)] += 1
        meta = dict(base["metadata"], has_assistant_text=with_text)
        states.append(
            {
                "id": row_id,
                "state": state,
                "state_sha256": hashlib.sha256(state.encode()).hexdigest(),
                "metadata": meta,
            }
        )
    digest = write_jsonl(ROOT / CONFIG["states"], states)
    manifest = {
        "rows": len(states),
        "with_assistant_text": {
            f"{k[0]}/{'text' if k[1] else 'none'}": v
            for k, v in sorted(changed.items())
        },
        "states_sha256": digest,
    }
    (ROOT / CONFIG["workload_dir"] / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
