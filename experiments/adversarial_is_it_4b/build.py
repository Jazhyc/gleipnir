"""Build the "is it a 4B thing?" sets (CPU only).

    PYTHONPATH=src:. python experiments/adversarial_is_it_4b/build.py \
        --source /path/to/multi-agent-wtf-mats-sprint/experiments/live-peers

Writes to the config's data directory, each in a student-prompt and a
teacher-prompt version with byte-identical trajectories:

- `calibration_*.jsonl`: honest and harmful strict-OOD rows, for each monitor's
  own flag lines;
- `framing_*.jsonl`: the lab_v03 framing readers under six posts;
- `liars_*.jsonl`: the 10 verified liars, their 62 recruiter controls, and each
  liar with its lies swapped for the truth (`swap_own`, `swap_all`).
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
from pathlib import Path

from gleipnir.adversarial import eval_sets
from gleipnir.adversarial.contract import MAX_TOKENS, join_prompt, render, split_prompt
from gleipnir.adversarial.monitors import MONITORS
from gleipnir.adversarial.multiagent import render_history, swap_posts

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text())
ROOT = eval_sets.ROOT
LAB_CONFIG = json.loads(
    (ROOT / "experiments/adversarial_lab_v03/config.json").read_text()
)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def instructions() -> dict[str, str]:
    texts = {
        name: (ROOT / path).read_text() for name, path in CONFIG["prompts"].items()
    }
    for name, text in texts.items():
        if sha(text) != CONFIG["prompt_sha256"][name]:
            raise ValueError(f"{name} prompt does not match its frozen SHA256")
    return texts


def lab_rows(name: str) -> list[dict]:
    path = ROOT / CONFIG["lab_data"] / f"{name}.jsonl"
    if eval_sets.file_sha256(path) != LAB_CONFIG["sha256"][name]:
        raise ValueError(f"{path} does not match its frozen lab_v03 SHA256")
    return [json.loads(line) for line in path.open()]


def with_head(row: dict, head: str, row_id: str | None = None, **extra) -> dict:
    trajectory = split_prompt(row["prompt"]).trajectory
    return {
        "id": row_id or row["id"],
        "prompt": join_prompt(head, trajectory),
        "metadata": {**row["metadata"], **extra},
    }


def calibration(texts, tokens) -> tuple[list[dict], list[dict], dict]:
    student = eval_sets.load("ood")
    teacher = {r["id"]: r for r in eval_sets.load("ood-teacher")}
    order = list(range(len(student)))
    random.Random(CONFIG["calibration"]["seed"]).shuffle(order)
    want = {0: CONFIG["calibration"]["honest"], 1: CONFIG["calibration"]["harmful"]}
    taken = {0: [], 1: []}
    skipped = collections.Counter()
    for index in order:
        row = student[index]
        label = eval_sets.label(row)
        if len(taken[label]) >= want[label]:
            if all(len(taken[k]) >= want[k] for k in want):
                break
            continue
        t_row = teacher[row["id"]]
        expected = join_prompt(texts["teacher"], split_prompt(row["prompt"]).trajectory)
        if t_row["prompt"] != expected:
            raise ValueError(
                f"{row['id']}: teacher prompt is not the student trajectory"
            )
        n = tokens(t_row["prompt"])
        if n > MAX_TOKENS:
            skipped[row["metadata"]["source_dataset"]] += 1
            continue
        taken[label].append((row, t_row, n))
    out_s, out_t = [], []
    for label in (0, 1):
        for rank, (row, t_row, n) in enumerate(taken[label]):
            extra = {"calibration_rank": rank, "teacher_tokens": n}
            out_s.append({**row, "metadata": {**row["metadata"], **extra}})
            out_t.append({**t_row, "metadata": {**t_row["metadata"], **extra}})
    sources = collections.Counter(
        (r["metadata"]["source_dataset"], eval_sets.label(r)) for r in out_s
    )
    summary = {
        "honest": len(taken[0]),
        "harmful": len(taken[1]),
        "skipped_over_limit_teacher": dict(skipped),
        "by_source_label": {f"{s}:{y}": n for (s, y), n in sorted(sources.items())},
    }
    return out_s, out_t, summary


def framing(texts, tokens) -> tuple[list[dict], list[dict], dict]:
    sets = {name: lab_rows(name) for name in ("framing", "framing_v2")}
    index = {
        (name, r["metadata"]["parent_id"], r["metadata"]["condition"]): r
        for name, rows in sets.items()
        for r in rows
    }
    readers = sorted({r["metadata"]["parent_id"] for r in sets["framing"]})
    subset = set(
        random.Random(CONFIG["subset"]["seed"]).sample(
            readers, CONFIG["subset"]["readers"]
        )
    )
    out_s, out_t = [], []
    for parent in readers:
        for condition, (name, original) in CONFIG["framing_conditions"].items():
            row = index[(name, parent, original)]
            extra = {
                "condition": condition,
                "lab_v03_id": row["id"],
                "subset": parent in subset,
            }
            row_id = f"{parent}::{condition}"
            out_s.append(with_head(row, texts["student"], row_id, **extra))
            if out_s[-1]["prompt"] != row["prompt"]:
                raise ValueError(f"{row_id}: student prompt changed")
            t_row = with_head(row, texts["teacher"], row_id, **extra)
            t_row["metadata"]["tokens"] = tokens(t_row["prompt"])
            out_t.append(t_row)
    summary = {"readers": len(readers), "subset": sorted(subset), "rows": len(out_s)}
    return out_s, out_t, summary


def world_posts(episode: Path) -> dict[int, dict]:
    posts = {}
    for line in (episode / "world.jsonl").open():
        event = json.loads(line)
        if event.get("type") == "board_post":
            posts[int(event["message"]["id"])] = event["message"]
    return posts


def swapped_texts(
    episode: Path, edits: dict[str, list]
) -> dict[int, tuple[str, str, str]]:
    """World post id -> (author, original text, swapped text)."""
    posts = world_posts(episode)
    result = {}
    for post_id, pairs in edits.items():
        post = posts[int(post_id)]
        text = post["text"]
        for old, new in pairs:
            if text.count(old) != 1:
                raise ValueError(f"{episode.name} post {post_id}: edit not found once")
            text = text.replace(old, new)
        result[int(post_id)] = (post["author"], post["text"], text)
    return result


QUARANTINE = re.compile(r"[^.;:!?]*quarantin[^.;:!?]*", re.I)
TRUTHFUL = re.compile(
    r"\b(not|no|without any)\b[^.;:!?]*quarantin"
    r"|quarantin[^.;:!?]*\b(does not|is not)\b",
    re.I,
)


def residual_claims(texts: list[str]) -> list[str]:
    """Clauses that still mention quarantine without negating it."""
    return [
        clause.strip()
        for text in texts
        for clause in QUARANTINE.findall(text)
        if not TRUTHFUL.search(clause)
    ]


def liars(texts, tokens, source: Path) -> tuple[list[dict], list[dict], dict]:
    lab = [
        r
        for r in lab_rows("lab")
        if r["metadata"]["condition"] in CONFIG["liar_groups"]
    ]
    swaps = json.loads((HERE / "liar_swaps.json").read_text())["episodes"]
    rows = list(lab)
    audit = {}
    for row in lab:
        if row["metadata"]["condition"] != "verified_liar":
            continue
        episode = source / row["metadata"]["task_id"]
        worker = row["metadata"]["worker"]
        history = json.loads((episode / f"{worker}-history.json").read_text())
        if render_history(history) != split_prompt(row["prompt"]).trajectory:
            raise ValueError(f"{row['id']}: history no longer renders to the lab row")
        table = swapped_texts(episode, swaps[episode.name])
        own_ids = {i for i, (author, _, _) in table.items() if author == worker}
        own_calls = sum(
            1
            for m in history
            if m.get("role") == "assistant"
            for p in m.get("content") or []
            if p.get("type") == "toolCall"
            and p.get("name") == "board_post"
            and any(p["arguments"].get("text") == table[i][1] for i in own_ids)
        )
        if own_calls != len(own_ids):
            raise ValueError(
                f"{row['id']}: {own_calls} own calls for {len(own_ids)} swaps"
            )
        for scope, ids in (("swap_own", own_ids), ("swap_all", set(table))):
            replacements = {i: table[i][1:] for i in ids}
            messages, count = swap_posts(history, replacements)
            rows.append(
                {
                    "id": f"{row['id']}::{scope}",
                    "prompt": join_prompt(texts["student"], render_history(messages)),
                    "metadata": {
                        **row["metadata"],
                        "condition": f"liar_{scope}",
                        "ground_truth": 0,
                        "parent_id": row["id"],
                        "swapped_posts": sorted(ids),
                        "replacements": count,
                    },
                }
            )
        own_new = [table[i][2] for i in own_ids]
        audit[row["id"]] = {
            "own_posts_swapped": len(own_ids),
            "all_posts_swapped": len(table),
            "residual_quarantine_clauses_in_own_posts": residual_claims(own_new),
        }
    out_s = [with_head(r, texts["student"]) for r in rows]
    out_t = [with_head(r, texts["teacher"]) for r in rows]
    for r in out_s + out_t:
        r["metadata"]["tokens"] = tokens(r["prompt"])
    summary = {
        "rows": len(out_s),
        "groups": collections.Counter(r["metadata"]["condition"] for r in out_s),
        "swap_audit": audit,
    }
    return out_s, out_t, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    texts = instructions()

    from transformers import AutoTokenizer

    spec = MONITORS["4b-v03"]
    tokenizer = AutoTokenizer.from_pretrained(spec.base, revision=spec.base_revision)

    def tokens(prompt: str) -> int:
        return len(
            tokenizer.encode(render(tokenizer, prompt), add_special_tokens=False)
        )

    out = ROOT / CONFIG["data"]
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"source": str(args.source.resolve()), "sets": {}, "sha256": {}}
    for name, builder in (
        ("calibration", lambda: calibration(texts, tokens)),
        ("framing", lambda: framing(texts, tokens)),
        ("liars", lambda: liars(texts, tokens, args.source.resolve())),
    ):
        student, teacher, summary = builder()
        manifest["sets"][name] = summary
        for prompt, rows in (("student", student), ("teacher", teacher)):
            path = out / f"{name}_{prompt}.jsonl"
            with path.open("w") as handle:
                for r in rows:
                    handle.write(json.dumps(r, ensure_ascii=False) + "\n")
            manifest["sha256"][f"{name}_{prompt}"] = eval_sets.file_sha256(path)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2, default=str)[:8000])


if __name__ == "__main__":
    main()
