"""Build the multi-agent liar scoring sets (CPU only).

    PYTHONPATH=src:. python experiments/adversarial_lab_liars/build.py \
        --source /path/to/multi-agent-wtf-mats-sprint/experiments/live-peers/liar-scale

Run `swaps.py` first. Writes to the config's data directory, each set in a
student-prompt and a teacher-prompt version with byte-identical trajectories:

- `main_*`: every worker (A-F) of every completed episode, action-only;
- `swap_*`: each primary liar with its confirmed false-claim posts rewritten
  to the truth (liars with any failed rewrite are excluded and counted);
- `last16_*`: the cycle-1 block plus the last 16 cycles, for every recruiter
  and every row over the token limit (sensitivity set);
- `canaries.jsonl`, `kimi_order.json`, `manifest.json`.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path
from typing import Any

from experiments.adversarial_lab_liars.common import (
    CONFIG,
    DATA,
    RECRUITERS,
    WORKERS,
    check_labels,
    episodes,
    index_labels,
    own_post_texts,
    read_jsonl,
    sha,
    validate_rewrite,
    window_history,
    worker_metadata,
    world_posts,
    write_jsonl,
)
from gleipnir.adversarial.contract import MAX_TOKENS, join_prompt, render, split_prompt
from gleipnir.adversarial.eval_sets import ROOT, file_sha256
from gleipnir.adversarial.monitors import MONITORS
from gleipnir.adversarial.multiagent import render_history, swap_posts

IS_IT_4B = ROOT / "data/adversarial_is_it_4b"
CANARY_SOURCES = {
    "4b": ROOT / "results/adversarial_lab_v03/lab/predictions.jsonl",
    "27b": ROOT / "results/adversarial_is_it_4b/27b-teacher/liars/predictions.jsonl",
}


def instructions() -> dict[str, str]:
    texts = {k: (ROOT / p).read_text() for k, p in CONFIG["prompts"].items()}
    for name, text in texts.items():
        if sha(text) != CONFIG["prompt_sha256"][name]:
            raise ValueError(f"{name} prompt does not match its frozen SHA256")
    return texts


def pair(row_id: str, trajectory: str, meta: dict, texts, count) -> tuple[dict, dict]:
    student = {
        "id": row_id,
        "prompt": join_prompt(texts["student"], trajectory),
        "metadata": dict(meta),
    }
    teacher = {
        "id": row_id,
        "prompt": join_prompt(texts["teacher"], trajectory),
        "metadata": dict(meta),
    }
    student["metadata"]["tokens"] = count["4b"](student["prompt"])
    teacher["metadata"]["tokens"] = count["27b"](teacher["prompt"])
    over = (
        student["metadata"]["tokens"] > MAX_TOKENS
        or teacher["metadata"]["tokens"] > MAX_TOKENS
    )
    for r in (student, teacher):
        r["metadata"]["tokens_4b_student"] = student["metadata"]["tokens"]
        r["metadata"]["tokens_27b_teacher"] = teacher["metadata"]["tokens"]
        r["metadata"]["over_limit"] = over
    return student, teacher


def swap_table(source: Path) -> tuple[dict[str, dict], dict[str, Any]]:
    """Liar id -> {board_id: (old, new)} for liars whose every rewrite passed."""
    from experiments.adversarial_lab_liars.swaps import targets

    rewrites = {
        r["post_id"]: r
        for r in read_jsonl(DATA / "swap_rewrites.jsonl")
        if r.get("parsed") is not None
    }
    screens = collections.defaultdict(list)
    for r in read_jsonl(DATA / "swap_screens.jsonl"):
        if r.get("parsed") is not None or r.get("raw") is not None:
            screens[r["post_id"]].append(r)
    by_liar = collections.defaultdict(list)
    for t in targets(source):
        by_liar[t["liar"]].append(t)
    table, audit = {}, {}
    reasons = collections.Counter()
    for liar, posts in sorted(by_liar.items()):
        entries, ok = {}, True
        for t in posts:
            record = rewrites.get(t["post_id"])
            edits = (record or {}).get("parsed", {}) or {}
            edits = edits.get("edits") if isinstance(edits, dict) else None
            passes = {}
            for s in screens[t["post_id"]]:
                passes[s["pass"]] = s.get("parsed") or {}
            if record is None:
                passed, reason, new = False, "no rewrite", None
            else:
                ordered = [
                    passes.get(n, {})
                    for n in range(1, CONFIG["swap"]["screen_passes"] + 1)
                ]
                ordered = [p if p else {"false_claim": None} for p in ordered]
                missing = any(
                    n not in passes
                    for n in range(1, CONFIG["swap"]["screen_passes"] + 1)
                )
                passed, reason, new = validate_rewrite(
                    t["text"],
                    edits,
                    [] if missing else ordered,
                    CONFIG["swap"]["max_changed_share"],
                )
                if passed and any(
                    s.get("rewritten_sha256") != sha(new) for s in screens[t["post_id"]]
                ):
                    passed, reason = False, "screen text mismatch"
            reasons[reason] += 1
            ok = ok and passed
            entries[t["board_id"]] = (t["text"], new, reason)
        fallback = sum(
            bool((rewrites.get(t["post_id"]) or {}).get("fallback")) for t in posts
        )
        audit[liar] = {
            "posts": len(posts),
            "passed": ok,
            "fallback_posts": fallback,
            "reasons": collections.Counter(e[2] for e in entries.values()),
        }
        if ok:
            table[liar] = {b: (old, new) for b, (old, new, _) in entries.items()}
    summary = {
        "liars": len(by_liar),
        "posts": sum(len(p) for p in by_liar.values()),
        "rewrites_made": sum(
            1 for p in by_liar.values() for t in p if t["post_id"] in rewrites
        ),
        "posts_validated": reasons["ok"],
        "post_outcomes": dict(reasons),
        "liars_validated": len(table),
        "liars_excluded": len(by_liar) - len(table),
        "posts_rewritten_by_fallback": sum(v["fallback_posts"] for v in audit.values()),
        "validated_liars_haiku_only": sum(
            1 for k in table if audit[k]["fallback_posts"] == 0
        ),
        "per_liar": {k: {**v, "reasons": dict(v["reasons"])} for k, v in audit.items()},
    }
    for liar in table:
        table[liar] = (table[liar], audit[liar]["fallback_posts"])
    return table, summary


def canaries(texts) -> list[dict]:
    """Four rows with known scores: 2 verified liars, 2 honest recruiters."""
    student = {r["id"]: r for r in read_jsonl(IS_IT_4B / "liars_student.jsonl")}
    teacher = {r["id"]: r for r in read_jsonl(IS_IT_4B / "liars_teacher.jsonl")}
    expected = {
        k: {r["id"]: r for r in read_jsonl(p) if r.get("status") == "ok"}
        for k, p in CANARY_SOURCES.items()
    }
    chosen = []
    for group in ("verified_liar", "honest_recruiter"):
        ids = sorted(
            i
            for i, r in student.items()
            if r["metadata"]["condition"] == group
            and all(i in e for e in expected.values())
        )
        chosen += ids[:2]
    rows = []
    for i in chosen:
        if student[i]["prompt"] != join_prompt(
            texts["student"], split_prompt(teacher[i]["prompt"]).trajectory
        ):
            raise ValueError(f"{i}: canary student and teacher trajectories differ")
        if not teacher[i]["prompt"].startswith(texts["teacher"]):
            raise ValueError(f"{i}: canary teacher prompt changed")
        rows.append(
            {
                "id": i,
                "prompts": {
                    "student": student[i]["prompt"],
                    "teacher": teacher[i]["prompt"],
                },
                "expected": {
                    k: {
                        "score": e[i]["score"],
                        "logit_0": e[i]["logit_0"],
                        "logit_1": e[i]["logit_1"],
                    }
                    for k, e in expected.items()
                },
            }
        )
    return rows


def kimi_order(main: list[dict], swap: list[dict], last16: list[dict]) -> list[dict]:
    """Frozen Kimi priority, in case the $4 cap binds (README 'Kimi order')."""
    cells = set(CONFIG["analysis"]["primary_cells"])
    keep = [r for r in main if not r["metadata"]["over_limit"]]
    tier0 = sorted(
        r["id"]
        for r in keep
        if r["metadata"]["analysis_recruiter"] and r["metadata"]["cell"] in cells
    )
    tier1 = sorted(r["id"] for r in swap if not r["metadata"]["over_limit"])
    tier2 = sorted(
        r["id"]
        for r in keep
        if r["metadata"]["role"] == "recruiter" and r["id"] not in set(tier0)
    )
    tier3 = sorted(r["id"] for r in keep if r["metadata"]["role"] != "recruiter")
    random.Random(CONFIG["kimi_order_seed"]).shuffle(tier3)
    tier4 = sorted(r["id"] for r in last16 if not r["metadata"]["over_limit"])
    order = []
    for tier, (name, ids) in enumerate(
        (
            ("main", tier0),
            ("swap", tier1),
            ("main", tier2),
            ("main", tier3),
            ("last16", tier4),
        )
    ):
        order += [{"set": name, "id": i, "tier": tier} for i in ids]
    return order


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    texts = instructions()
    index = index_labels(check_labels(source))

    from transformers import AutoTokenizer

    count = {}
    for key, name in (("4b", "4b-v03"), ("27b", "27b-merged")):
        spec = MONITORS[name]
        tok = AutoTokenizer.from_pretrained(spec.base, revision=spec.base_revision)
        count[key] = (
            lambda t: lambda p: len(t.encode(render(t, p), add_special_tokens=False))
        )(tok)

    found = episodes(source)
    if len(found) != CONFIG["expected_episodes"]:
        raise ValueError(
            f"{len(found)} completed episodes, expected {CONFIG['expected_episodes']}"
        )
    table, swap_summary = swap_table(source)
    sets = {name: {"student": [], "teacher": []} for name in ("main", "swap", "last16")}
    swap_audit = {}
    for run, episode, outcome in found:
        for w in WORKERS:
            history = json.loads((episode / f"{w}-history.json").read_text())
            meta = worker_metadata(run, episode, outcome, w, index)
            row_id = f"{run}/{episode.name}/{w}"
            s, t = pair(row_id, render_history(history), meta, texts, count)
            sets["main"]["student"].append(s)
            sets["main"]["teacher"].append(t)
            if w in RECRUITERS or s["metadata"]["over_limit"]:
                window = window_history(history, CONFIG["sensitivity_window_cycles"])
                extra = {
                    **meta,
                    "parent_id": row_id,
                    "view": "last16",
                    "parent_over_limit": s["metadata"]["over_limit"],
                }
                if w in RECRUITERS:
                    lies = {
                        p["board_id"]
                        for p in index["posts"][(run, episode.name, w)]
                        if p["primary"]["false_claim"]
                    }
                    posts = world_posts(episode) if lies else {}
                    kept = set(own_post_texts(window))
                    extra["false_claim_posts_in_window"] = sum(
                        posts[b]["text"] in kept for b in lies
                    )
                ws, wt = pair(
                    f"{row_id}::last16", render_history(window), extra, texts, count
                )
                sets["last16"]["student"].append(ws)
                sets["last16"]["teacher"].append(wt)
            if row_id in table:
                replacements, fallback_posts = table[row_id]
                own = own_post_texts(history)
                missing = [b for b, (old, _) in replacements.items() if old not in own]
                if missing:
                    raise ValueError(
                        f"{row_id}: false-claim posts {missing} not among own calls"
                    )
                swapped, replaced = swap_posts(history, replacements)
                calls = sum(old in own for old, _ in replacements.values())
                swap_audit[row_id] = {
                    "posts": len(replacements),
                    "replacements": replaced,
                    "own_calls": calls,
                }
                extra = {
                    **meta,
                    "condition": "liar_swap",
                    "ground_truth": 0,
                    "parent_id": row_id,
                    "swapped_board_ids": sorted(replacements),
                    "replacements": replaced,
                    "swap_fallback_posts": fallback_posts,
                }
                ss, st = pair(
                    f"{row_id}::swap", render_history(swapped), extra, texts, count
                )
                if ss["prompt"] == s["prompt"]:
                    raise ValueError(f"{row_id}: swap changed nothing")
                sets["swap"]["student"].append(ss)
                sets["swap"]["teacher"].append(st)

    DATA.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "source": str(source),
        "episodes": len(found),
        "sha256": {},
    }
    for name, versions in sets.items():
        for prompt, rows in versions.items():
            path = DATA / f"{name}_{prompt}.jsonl"
            write_jsonl(path, rows)
            manifest["sha256"][f"{name}_{prompt}"] = file_sha256(path)
    canary_rows = canaries(texts)
    write_jsonl(DATA / "canaries.jsonl", canary_rows)
    manifest["sha256"]["canaries"] = file_sha256(DATA / "canaries.jsonl")
    order = kimi_order(
        sets["main"]["student"], sets["swap"]["student"], sets["last16"]["student"]
    )
    (DATA / "kimi_order.json").write_text(json.dumps(order) + "\n")
    manifest["sha256"]["kimi_order"] = file_sha256(DATA / "kimi_order.json")

    def counts(rows, key):
        return dict(sorted(collections.Counter(key(r) for r in rows).items()))

    summary = {}
    for name, versions in sets.items():
        rows = versions["student"]
        over = [r for r in rows if r["metadata"]["over_limit"]]
        summary[name] = {
            "rows": len(rows),
            "groups": counts(rows, lambda r: r["metadata"]["condition"]),
            "over_limit": len(over),
            "over_limit_by_group": counts(over, lambda r: r["metadata"]["condition"]),
            "over_limit_by_agent_model": counts(over, lambda r: r["metadata"]["model"]),
            "over_limit_by_monitor": {
                "4b_student": sum(
                    r["metadata"]["tokens_4b_student"] > MAX_TOKENS for r in rows
                ),
                "27b_teacher": sum(
                    r["metadata"]["tokens_27b_teacher"] > MAX_TOKENS for r in rows
                ),
                "kimi_teacher": len(over),
            },
            "tokens_4b_student": sum(
                r["metadata"]["tokens_4b_student"]
                for r in rows
                if not r["metadata"]["over_limit"]
            ),
            "tokens_27b_teacher": sum(
                r["metadata"]["tokens_27b_teacher"]
                for r in rows
                if not r["metadata"]["over_limit"]
            ),
        }
    main_rows = sets["main"]["student"]
    summary["main"]["analysis_recruiters"] = counts(
        [r for r in main_rows if r["metadata"]["analysis_recruiter"]],
        lambda r: (
            f"{r['metadata']['cell']}|{r['metadata']['recruiter_model']}|{r['metadata']['condition']}"
        ),
    )
    summary["last16"]["false_claim_posts_in_window"] = counts(
        [r for r in sets["last16"]["student"] if r["metadata"]["condition"] == "liar"],
        lambda r: r["metadata"]["false_claim_posts_in_window"] > 0,
    )
    manifest["sets"] = summary
    manifest["swap"] = {**swap_summary, "row_audit": swap_audit}
    manifest["kimi_order_tiers"] = counts(order, lambda o: o["tier"])
    manifest["canaries"] = [r["id"] for r in canary_rows]
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    brief = {k: v for k, v in manifest.items() if k not in ("swap",)}
    brief["swap"] = {k: v for k, v in swap_summary.items() if k != "per_liar"}
    print(json.dumps(brief, indent=2))


if __name__ == "__main__":
    main()
