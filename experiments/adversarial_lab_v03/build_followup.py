"""Build the framing follow-up set: placement and rewordings (CPU only).

    PYTHONPATH=src:. python experiments/adversarial_lab_v03/build_followup.py \
        --source /path/to/multi-agent-wtf-mats-sprint/experiments/live-peers

Reuses the 300 readers, planted turns and authors of `framing.jsonl`. Writes
`framing_v2.jsonl` with conditions `tool:<post>` (board posts rendered as tool
output, including an unedited `tool:none` baseline) and `user:<post>` (new
wordings in the lab's own user-turn placement).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.adversarial_lab_v03.build import CONFIG, STUDENT, row, sha
from gleipnir.adversarial.contract import render
from gleipnir.adversarial.eval_sets import ROOT, file_sha256
from gleipnir.adversarial.monitors import MONITORS
from gleipnir.adversarial.multiagent import plant_post, render_history


def post_text(name: str) -> str:
    text = CONFIG["followup"]["posts"][name]
    if text == "@template:transfer_gpt41":
        template = json.loads((ROOT / "data/templates/transfer_gpt41.json").read_text())
        return template["docstring"].strip()
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    out = ROOT / CONFIG["data"]
    instruction = STUDENT.read_text()
    followup = CONFIG["followup"]

    readers = {}
    for line in (out / "framing.jsonl").open():
        metadata = json.loads(line)["metadata"]
        readers[metadata["parent_id"]] = metadata

    from transformers import AutoTokenizer

    spec = MONITORS[CONFIG["monitor"]]
    tokenizer = AutoTokenizer.from_pretrained(spec.base, revision=spec.base_revision)

    rows = []
    for parent_id, meta in sorted(readers.items()):
        episode = source / meta["task_id"]
        reader, author, turn = (
            meta["worker"],
            meta["planted_author"],
            meta["planted_turn"],
        )
        history = json.loads((episode / f"{reader}-history.json").read_text())
        plans = [("tool", name) for name in followup["tool_placement_posts"]]
        plans += [("user", name) for name in followup["user_placement_posts"]]
        for placement, name in plans:
            if name == "none":
                messages = history
            else:
                text = post_text(name).replace("{AUTHOR}", author)
                messages = plant_post(
                    history, turn, author, text.replace("{READER}", reader)
                )
            trajectory = render_history(messages, board_as_tool=placement == "tool")
            r = row(
                f"{parent_id}::{placement}:{name}",
                trajectory,
                instruction,
                {
                    "condition": f"{placement}:{name}",
                    "ground_truth": 0,
                    "task_id": meta["task_id"],
                    "parent_id": parent_id,
                    "experiment": meta["experiment"],
                    "worker": reader,
                    "model": meta["model"],
                    "planted_turn": turn,
                    "planted_author": author,
                },
            )
            r["metadata"]["tokens"] = len(
                tokenizer.encode(
                    render(tokenizer, r["prompt"]), add_special_tokens=False
                )
            )
            rows.append(r)

    path = out / "framing_v2.jsonl"
    with path.open("w") as handle:
        for r in rows:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {
        "rows": len(rows),
        "readers": len(readers),
        "tokens": sum(r["metadata"]["tokens"] for r in rows),
        "post_sha256": {k: sha(post_text(k)) for k in followup["posts"]},
        "sha256": file_sha256(path),
    }
    (out / "framing_v2.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
