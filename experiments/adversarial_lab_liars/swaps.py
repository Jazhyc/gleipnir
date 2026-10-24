"""Truth-swap rewrites for every primary liar (Claude Haiku 5.5 via OpenRouter).

    PYTHONPATH=src:. python experiments/adversarial_lab_liars/swaps.py \
        --source /path/to/multi-agent-wtf-mats-sprint/experiments/live-peers/liar-scale

For each confirmed (primary) false-claim post by a primary liar: one rewrite
call with `rewrite_prompt.md`, then the lab's lie screen
(`labels_screen_prompt.md`) on the rewritten text, twice. Resumable; records
go to the ignored data directory. Validation (`common.validate_rewrite`) is
applied by `build.py`. Cap: $0.50 for rewrites and screens together.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from experiments.adversarial_lab_liars.common import (
    CONFIG,
    DATA,
    HERE,
    check_labels,
    episodes,
    index_labels,
    openrouter_balance,
    openrouter_key,
    read_jsonl,
    sha,
    world_posts,
)

SPEC = CONFIG["swap"]
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
REWRITE_SCHEMA = {
    "name": "rewrite",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "edits": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "old": {"type": "string"},
                        "new": {"type": "string"},
                    },
                    "required": ["old", "new"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["edits"],
        "additionalProperties": False,
    },
}
# The lab screen's schema and request settings (liar-scale src/labels_screen.mjs).
SCREEN_SCHEMA = {
    "name": "screen",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "false_claim": {"type": "string", "enum": ["yes", "no"]},
            "retraction": {"type": "string", "enum": ["yes", "no"]},
        },
        "required": ["false_claim", "retraction"],
        "additionalProperties": False,
    },
}


def fill(template: str, author: str, cycle: Any, text: str) -> str:
    return (
        template.replace("{{FACT_SHEET}}", SPEC["fact_sheet"])
        .replace("{{AUTHOR}}", author)
        .replace("{{CYCLE}}", str(cycle))
        .replace("{{TEXT}}", text)
    )


def body(prompt: str, schema: dict, max_tokens: int) -> dict[str, Any]:
    # Haiku 5.5 on OpenRouter rejects temperature under require_parameters
    # (liar-scale Amendment 4), so none is sent and reasoning is disabled.
    return {
        "model": SPEC["model"],
        "max_tokens": max_tokens,
        "reasoning": {"enabled": False},
        "provider": {
            "require_parameters": True,
            "allow_fallbacks": False,
            "only": [SPEC["provider_only"]],
        },
        "response_format": {"type": "json_schema", "json_schema": schema},
        "messages": [{"role": "user", "content": prompt}],
        "usage": {"include": True},
    }


def fallback_body(prompt: str) -> dict[str, Any]:
    spec = SPEC["fallback"]
    return {
        "model": spec["model"],
        "max_tokens": spec["max_tokens"],
        "reasoning": {"effort": spec["reasoning_effort"]},
        "provider": {"require_parameters": True, "allow_fallbacks": False},
        "response_format": {"type": "json_schema", "json_schema": REWRITE_SCHEMA},
        "messages": [{"role": "user", "content": prompt}],
        "usage": {"include": True},
    }


def parse_json(content: str) -> dict[str, Any] | None:
    try:
        value = json.loads(content)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def call(payload: dict[str, Any], key: str, post=None) -> dict[str, Any]:
    """One logical call: HTTP retries on overload, retries on empty output."""
    import requests

    post = post or requests.post
    spent, attempts, last = 0.0, 0, {}
    for _ in range(SPEC["empty_output_attempts"]):
        for http_try in range(4):
            response = post(
                ENDPOINT,
                headers={"Authorization": f"Bearer {key}"},
                json=payload,
                timeout=120,
            )
            if response.status_code < 400:
                break
            if http_try == 3 or response.status_code not in {
                408,
                429,
                500,
                502,
                503,
                504,
            }:
                return {
                    "error": f"HTTP {response.status_code}",
                    "cost": spent,
                    "attempts": attempts,
                }
            time.sleep(2 * (http_try + 1))
        data = response.json()
        attempts += 1
        spent += float((data.get("usage") or {}).get("cost") or 0.0)
        content = ((data.get("choices") or [{}])[0].get("message") or {}).get(
            "content"
        ) or ""
        choice = (data.get("choices") or [{}])[0]
        last = {
            "raw": content,
            "provider": data.get("provider"),
            "model": data.get("model"),
            "finish_reason": choice.get("finish_reason"),
        }
        if parse_json(content) is not None:
            break
    return {
        **last,
        "parsed": parse_json(last.get("raw", "")),
        "cost": spent,
        "attempts": attempts,
    }


def targets(source: Path) -> list[dict[str, Any]]:
    """Every primary false-claim post by every primary liar, with world text."""
    index = index_labels(check_labels(source))
    out = []
    for run, episode, _outcome in episodes(source):
        posts = None
        for author in "BC":
            label = index["recruiters"][(run, episode.name, author, "primary")]
            if label["label"] != "liar":
                continue
            posts = posts or world_posts(episode)
            for p in index["posts"][(run, episode.name, author)]:
                if p["primary"]["false_claim"]:
                    world = posts[int(p["board_id"])]
                    if world["author"] != author:
                        raise ValueError(f"{p['post_id']}: author mismatch")
                    out.append(
                        {
                            "post_id": p["post_id"],
                            "liar": f"{run}/{episode.name}/{author}",
                            "board_id": int(p["board_id"]),
                            "author": author,
                            "cycle": world["cycle"],
                            "text": world["text"],
                        }
                    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    rewrite_template = (HERE / "rewrite_prompt.md").read_text()
    screen_template = (source / "src" / "labels_screen_prompt.md").read_text()
    if sha(screen_template) != CONFIG["prompt_sha256"]["lie_screen"]:
        raise ValueError("lab screen prompt does not match its frozen SHA256")
    todo = targets(source)
    DATA.mkdir(parents=True, exist_ok=True)
    rewrites_path, screens_path = (
        DATA / "swap_rewrites.jsonl",
        DATA / "swap_screens.jsonl",
    )
    rewrites = {
        r["post_id"]: r
        for r in read_jsonl(rewrites_path)
        if r.get("parsed") is not None
    }
    screens = {
        (r["post_id"], r["pass"])
        for r in read_jsonl(screens_path)
        if r.get("parsed") is not None
    }
    spent = [
        sum(r.get("cost") or 0 for r in read_jsonl(rewrites_path))
        + sum(r.get("cost") or 0 for r in read_jsonl(screens_path))
    ]
    key = openrouter_key()
    before = openrouter_balance(key)
    print(
        f"posts {len(todo)}; balance before ${before:.4f}; spent so far ${spent[0]:.4f}"
    )
    lock = threading.Lock()

    def append(path: Path, record: dict[str, Any]) -> None:
        with lock:
            spent[0] += record.get("cost") or 0.0
            with path.open("a") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    haiku_failed = {
        r["post_id"]
        for r in read_jsonl(rewrites_path)
        if r.get("provider") == "Anthropic" and r.get("parsed") is None
    }

    def rewrite(target):
        if target["post_id"] in rewrites or spent[0] >= SPEC["max_usd"]:
            return
        prompt = fill(
            rewrite_template, target["author"], target["cycle"], target["text"]
        )
        base = {
            "post_id": target["post_id"],
            "liar": target["liar"],
            "prompt_sha256": sha(rewrite_template),
            "text_sha256": sha(target["text"]),
        }
        if target["post_id"] not in haiku_failed:
            record = {
                **base,
                "fallback": False,
                **call(body(prompt, REWRITE_SCHEMA, SPEC["max_tokens"]), key),
            }
            append(rewrites_path, record)
            if record.get("parsed") is not None:
                rewrites[target["post_id"]] = record
                return
        # Haiku gave no parseable rewrite (content filter or empty output):
        # the same prompt goes to the frozen fallback model once (Amendment 1).
        record = {**base, "fallback": True, **call(fallback_body(prompt), key)}
        append(rewrites_path, record)
        if record.get("parsed") is not None:
            rewrites[target["post_id"]] = record

    with ThreadPoolExecutor(SPEC["concurrency"]) as pool:
        list(pool.map(rewrite, todo))

    from experiments.adversarial_lab_liars.common import apply_edits

    jobs = []
    for target in todo:
        record = rewrites.get(target["post_id"])
        if record is None:
            continue
        try:
            new = apply_edits(target["text"], record["parsed"].get("edits"))
        except (ValueError, AttributeError, TypeError):
            continue  # fails validation without a screen
        for n in range(1, SPEC["screen_passes"] + 1):
            if (target["post_id"], n) not in screens:
                jobs.append((target, new, n))

    def screen(job):
        target, new, n = job
        if spent[0] >= SPEC["max_usd"]:
            return
        prompt = fill(screen_template, target["author"], target["cycle"], new)
        out = call(body(prompt, SCREEN_SCHEMA, SPEC["screen_max_tokens"]), key)
        append(
            screens_path,
            {
                "post_id": target["post_id"],
                "pass": n,
                "rewritten_sha256": sha(new),
                **out,
            },
        )

    with ThreadPoolExecutor(SPEC["concurrency"]) as pool:
        list(pool.map(screen, jobs))
    after = openrouter_balance(key)
    print(
        json.dumps(
            {
                "posts": len(todo),
                "rewrites_parsed": len(rewrites),
                "screen_jobs": len(jobs),
                "spent_total": round(spent[0], 6),
                "stopped_at_cap": spent[0] >= SPEC["max_usd"],
                "balance_after": round(after, 4),
            }
        )
    )


if __name__ == "__main__":
    main()
