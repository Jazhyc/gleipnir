"""Freeze the replay session, check schedules and a token-length index."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.b200_prefix_replay.session import (
    SENTINEL,
    build,
    content,
    policies,
)
from gleipnir.data.monitoring import file_hash, read_rows, write_json

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/teacher-disagreement-delta000-sdpa04/training.jsonl"
TOKENIZER = ROOT / "data/judge_injection_continuation/tokenizer"
OUT = ROOT / "data/b200_prefix_replay"
LIMIT = 1_040_000  # below the 1,048,576-token YaRN limit, leaving template room


def main() -> None:
    from transformers import AutoTokenizer

    if OUT.exists():
        raise ValueError("replay session already prepared")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    template = (
        tokenizer.apply_chat_template(
            [{"role": "user", "content": SENTINEL}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        + "Prediction:"
    )

    def count(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    rows = [r for r in read_rows(SOURCE) if r["raw_source"] == "gloom"]
    session = build(rows, count, LIMIT - count(template))
    session["template"] = template
    session["policies"] = policies(session["steps"])
    # Cumulative token estimate per step (piecewise; boundary merges are tiny).
    running, cumulative = count(session["header"]), []
    for step in session["steps"]:
        running += count(step["block"])
        cumulative.append(running)
    session["cumulative_tokens_estimate"] = cumulative
    last = session["policies"]["every_action"][-1]
    session["last_check_tokens"] = count(
        template.replace(SENTINEL, content(session, last))
    )
    OUT.mkdir(parents=True)
    path = OUT / "session.json"
    path.write_text(json.dumps(session) + "\n")
    write_json(
        OUT / "manifest.json",
        {
            "source": str(SOURCE.relative_to(ROOT)),
            "source_sha256": file_hash(SOURCE),
            "gloom_trajectories": len({s["source_index"] for s in session["steps"]}),
            "steps": len(session["steps"]),
            "checks": {k: len(v) for k, v in session["policies"].items()},
            "last_check_tokens": session["last_check_tokens"],
            "session_sha256": file_hash(path),
            "template_sha256": hashlib.sha256(template.encode()).hexdigest(),
            "kind": "synthetic chained Gloom steps; unlabelled systems workload",
        },
    )
    print(json.dumps(json.loads((OUT / "manifest.json").read_text()), indent=1))


if __name__ == "__main__":
    main()
