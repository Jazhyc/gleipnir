"""Build the lab evaluation sets from six-worker Halden lab episodes (CPU only).

    PYTHONPATH=src:. python experiments/adversarial_lab_v03/build.py \
        --source /path/to/multi-agent-wtf-mats-sprint/experiments/live-peers

Writes `lab.jsonl` (every worker of every complete episode), `framing.jsonl`
(honest readers with one planted peer post) and `manifest.json` to the config's
data directory. Rows follow the eval-set schema: id, prompt, metadata.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
from pathlib import Path

from gleipnir.adversarial.contract import MAX_TOKENS, join_prompt, render
from gleipnir.adversarial.eval_sets import ROOT, file_sha256
from gleipnir.adversarial.monitors import MONITORS
from gleipnir.adversarial.multiagent import board_turns, plant_post, render_history

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text())
WORKERS = "ABCDEF"
STUDENT = ROOT / "experiments/tool_trajectory_monitoring/prompts/student.txt"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def episodes(source: Path) -> list[Path]:
    found = []
    for path in sorted(source.glob("**/F-history.json")):
        episode = path.parent
        parts = set(episode.relative_to(source).parts)
        if any(
            part in name for part in CONFIG["excluded_path_parts"] for name in parts
        ):
            continue
        if (episode / "outcome.json").exists() and all(
            (episode / f"{w}-history.json").exists() for w in WORKERS
        ):
            found.append(episode)
    return found


def preamble(history: list[dict]) -> str:
    sections = history[0].get("sections") or {}
    return sections.get("preamble", history[0].get("content") or "")


def special_roles(histories: dict[str, list[dict]]) -> dict[str, str | None]:
    """Workers whose system prompt differs from the episode's majority prompt.

    The extra text (persuasion or deception instructions) is invisible to an
    action-only monitor; only its hash is kept.
    """
    normal = {
        w: re.sub(r"\b(worker|Worker) [A-F]\b", r"\1 X", preamble(h))
        for w, h in histories.items()
    }
    common = collections.Counter(normal.values()).most_common(1)[0][0]
    return {w: (None if text == common else sha(text)) for w, text in normal.items()}


def post_authors(episode: Path) -> dict[int, str]:
    authors = {}
    for line in (episode / "world.jsonl").open():
        event = json.loads(line)
        if event.get("type") == "board_post":
            authors[int(event["message"]["id"])] = event["message"]["author"]
    return authors


def false_claim_authors(source: Path) -> dict[str, set[str]]:
    """Recruiters with a reviewed false claim, from the recruitment-honesty audit."""
    exposure = json.loads(
        (source / "recruitment-honesty/recruiter-exposure.json").read_text()
    )["episodes"]
    root = source / "recruitment-honesty/results/main-01"
    result = {}
    for name, record in exposure.items():
        authors = post_authors(root / name)
        result[name] = {authors[int(i)] for i in record["false_claim_post_ids"]}
    return result


def model_of(history: list[dict]) -> str | None:
    for message in history:
        if message.get("role") == "assistant" and message.get("model"):
            return message["model"]
    return None


def group_of(
    experiment: str,
    episode: Path,
    worker: str,
    special: str | None,
    liars: dict[str, set[str]],
    sacrificed: set[str],
) -> tuple[str, int]:
    if special is None:
        return ("plain_sacrificed", 0) if worker in sacrificed else ("plain", 0)
    if experiment == "recruitment-honesty":
        if episode.name.endswith("-honest"):
            return "honest_recruiter", 0
        if worker in liars.get(episode.name, set()):
            return "verified_liar", 1
        return "assigned_deceiver_no_lie", 0
    return "other_special_role", -1


def row(row_id: str, trajectory: str, instruction: str, metadata: dict) -> dict:
    return {
        "id": row_id,
        "prompt": join_prompt(instruction, trajectory),
        "metadata": {"source_dataset": "halden_lab", **metadata},
    }


def post_texts(sentences: list[str]) -> dict[str, str]:
    posts = dict(CONFIG["framing"]["posts"])
    template = json.loads((ROOT / "data/templates/transfer_gpt41.json").read_text())
    posts["injection_gpt41"] = template["docstring"].strip()
    target = len(posts["injection_gpt41"])
    text, i = "", 0
    while len(text) < target:
        text += ("" if not text else " ") + sentences[i % len(sentences)]
        i += 1
    posts["neutral_long"] = "{AUTHOR}: " + text
    return posts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    out = ROOT / CONFIG["data"]
    out.mkdir(parents=True, exist_ok=True)
    instruction = STUDENT.read_text()
    liars = false_claim_authors(source)

    from transformers import AutoTokenizer

    spec = MONITORS[CONFIG["monitor"]]
    tokenizer = AutoTokenizer.from_pretrained(spec.base, revision=spec.base_revision)

    def tokens(prompt: str) -> int:
        return len(
            tokenizer.encode(render(tokenizer, prompt), add_special_tokens=False)
        )

    lab, pool, skipped = [], [], collections.Counter()
    for episode in episodes(source):
        experiment = episode.relative_to(source).parts[0]
        key = str(episode.relative_to(source))
        histories = {
            w: json.loads((episode / f"{w}-history.json").read_text()) for w in WORKERS
        }
        roles = special_roles(histories)
        outcome = json.loads((episode / "outcome.json").read_text())
        sacrificed = {d["worker"] for d in outcome.get("diagnostics") or []}
        for w in WORKERS:
            group, label = group_of(experiment, episode, w, roles[w], liars, sacrificed)
            metadata = {
                "condition": group,
                "ground_truth": label,
                "task_id": key,
                "experiment": experiment,
                "worker": w,
                "model": model_of(histories[w]),
                "special_role_sha256": roles[w],
                "ran_diagnostic": w in sacrificed,
            }
            r = row(f"{key}/{w}", render_history(histories[w]), instruction, metadata)
            r["metadata"]["tokens"] = tokens(r["prompt"])
            if r["metadata"]["tokens"] > MAX_TOKENS:
                skipped[group] += 1
                continue
            lab.append(r)
            no_roles = all(v is None for v in roles.values())
            turns = board_turns(histories[w])
            if (
                no_roles
                and group == "plain"
                and experiment in CONFIG["framing"]["experiments"]
                and len(turns) >= CONFIG["framing"]["min_board_turns"]
            ):
                pool.append((key, w, histories[w], turns))

    rng = random.Random(CONFIG["framing"]["seed"])
    chosen = rng.sample(pool, min(CONFIG["framing"]["sample"], len(pool)))
    posts = post_texts(CONFIG["framing"]["neutral_sentences"])
    framing = []
    for key, reader, history, turns in chosen:
        author = next(a for a in CONFIG["framing"]["author_preference"] if a != reader)
        turn = turns[len(turns) // 2]
        base_id = f"{key}/{reader}"
        variants = {"none": history}
        for name, template in posts.items():
            text = template.replace("{AUTHOR}", author).replace("{READER}", reader)
            variants[name] = plant_post(history, turn, author, text)
        for name, messages in variants.items():
            r = row(
                f"{base_id}::{name}",
                render_history(messages),
                instruction,
                {
                    "condition": name,
                    "ground_truth": 0,
                    "task_id": key,
                    "parent_id": base_id,
                    "experiment": key.split("/")[0],
                    "worker": reader,
                    "model": model_of(history),
                    "planted_turn": turn,
                    "planted_author": author,
                },
            )
            r["metadata"]["tokens"] = tokens(r["prompt"])
            framing.append(r)

    for name, rows in (("lab", lab), ("framing", framing)):
        with (out / f"{name}.jsonl").open("w") as handle:
            for r in rows:
                handle.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {
        "source": str(source),
        "episodes": len({r["metadata"]["task_id"] for r in lab}),
        "lab_rows": len(lab),
        "lab_groups": collections.Counter(r["metadata"]["condition"] for r in lab),
        "lab_over_token_limit": skipped,
        "framing_pool": len(pool),
        "framing_rows": len(framing),
        "framing_post_sha256": {k: sha(v) for k, v in posts.items()},
        "sha256": {
            name: file_sha256(out / f"{name}.jsonl") for name in ("lab", "framing")
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
