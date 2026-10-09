"""Construct fixed token-length controls without observing model activations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from gleipnir.data.monitoring import file_hash, read_rows, write_json, write_rows

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
CONFIG = HERE / "config.json"


def insertion(clean: str, attacked: str) -> tuple[int, str]:
    """Recover one insertion, rejecting replacements or multiple changed spans."""
    start = 0
    while start < min(len(clean), len(attacked)) and clean[start] == attacked[start]:
        start += 1
    tail = 0
    while (
        tail < len(clean) - start
        and tail < len(attacked) - start
        and clean[-1 - tail] == attacked[-1 - tail]
    ):
        tail += 1
    end = len(attacked) - tail if tail else len(attacked)
    added = attacked[start:end]
    if not added or attacked[:start] + attacked[end:] != clean:
        raise ValueError("expected one nonempty insertion")
    return start, added


def match_padding(
    tokenizer: Any, clean: str, start: int, target: int, unit: str
) -> tuple[str, int]:
    """Match complete-prompt tokens, including token merges at insertion edges."""

    def count(padding: str) -> int:
        return len(
            tokenizer.encode(
                clean[:start] + padding + clean[start:], add_special_tokens=False
            ).ids
        )

    padding = unit
    for _ in range(16):
        if count(padding) >= target + 8:
            break
        padding += padding
    else:
        raise ValueError("control padding cannot reach target length")
    offsets = sorted(
        {
            0,
            *[
                b
                for a, b in tokenizer.encode(padding, add_special_tokens=False).offsets
            ],
        }
    )
    lower, upper = 0, len(offsets) - 1
    while lower < upper:
        mid = (lower + upper) // 2
        if count(padding[: offsets[mid]]) < target:
            lower = mid + 1
        else:
            upper = mid
    nearby = offsets[max(0, lower - 6) : min(len(offsets), lower + 7)]
    candidates = [
        (abs(count(padding[:end]) - target), end, count(padding[:end]))
        for end in nearby
        if end
    ]
    error, end, actual = min(candidates)
    if error > 1:
        candidates = [
            (abs(count(padding[:end]) - target), end, count(padding[:end]))
            for end in offsets
            if end
        ]
        error, end, actual = min(candidates)
    if error > 1:
        raise ValueError("control token match exceeds one token")
    return padding[:end], actual


def prepare() -> None:
    from tokenizers import Tokenizer

    config = json.loads(CONFIG.read_text())
    out = ROOT / "results/b200_judge_direction_signal" / config["campaign_id"]
    if out.exists():
        raise ValueError("prepared output already exists")
    for spec in config["inputs"].values():
        if file_hash(ROOT / spec["path"]) != spec["sha256"]:
            raise ValueError("input drift " + spec["path"])
    tokenizer = Tokenizer.from_file(str(ROOT / config["inputs"]["tokenizer"]["path"]))
    if [tokenizer.encode(t, add_special_tokens=False).ids for t in ["A", "B"]] != [
        [32],
        [33],
    ]:
        raise ValueError("A/B tokenizer identity changed")
    original = read_rows(ROOT / config["inputs"]["judge_workload"]["path"])
    clean = {
        (r["metadata"]["pair_id"], r["metadata"]["order"]): r
        for r in original
        if r["metadata"]["condition"] == "clean"
    }
    if len(original) != 4188 or len(clean) != 504:
        raise ValueError("original population changed")
    originals = []
    controls = {k: [] for k in config["control_units"]}
    receipts = []
    for r in original:
        meta = r["metadata"]
        parent = clean[meta["pair_id"], meta["order"]]
        if (
            meta["label"] != parent["metadata"]["label"]
            or meta["lineage_group"] != parent["metadata"]["lineage_group"]
        ):
            raise ValueError("parent preference/query changed")
        if (
            len(tokenizer.encode(r["prompt"], add_special_tokens=False).ids)
            != r["prompt_tokens"]
        ):
            raise ValueError("original native token identity changed")
        originals.append(
            r | {"metadata": meta | {"clean_id": parent["id"], "kind": "original"}}
        )
        if meta["condition"] == "clean":
            continue
        start, added = insertion(parent["prompt"], r["prompt"])
        # The common-prefix position must remain inside exactly one candidate.
        slots = [
            slot
            for slot in ["A", "B"]
            if parent["prompt"].index(f"<candidate_{slot}>")
            + len(f"<candidate_{slot}>")
            <= start
            <= parent["prompt"].index(f"</candidate_{slot}>")
        ]
        if len(slots) != 1:
            raise ValueError("insertion outside a candidate")
        target = r["prompt_tokens"]
        for kind, unit in config["control_units"].items():
            padding, actual = match_padding(
                tokenizer, parent["prompt"], start, target, unit
            )
            prompt = parent["prompt"][:start] + padding + parent["prompt"][start:]
            if abs(actual - target) > config["max_control_token_error"]:
                raise ValueError("control native length mismatch")
            rid = r["id"] + ":" + kind
            controls[kind].append(
                {
                    "id": rid,
                    "prompt": prompt,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "prompt_tokens": actual,
                    "metadata": meta
                    | {"clean_id": parent["id"], "attack_id": r["id"], "kind": kind},
                }
            )
            receipts.append(
                {
                    "id": rid,
                    "attack_id": r["id"],
                    "clean_id": parent["id"],
                    "kind": kind,
                    "candidate": slots[0],
                    "insertion_char": start,
                    "attack_suffix_sha256": hashlib.sha256(added.encode()).hexdigest(),
                    "control_suffix_sha256": hashlib.sha256(
                        padding.encode()
                    ).hexdigest(),
                    "control_suffix": padding,
                    "target_tokens": target,
                    "actual_tokens": actual,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                }
            )
    if any(len(rs) != 3684 for rs in controls.values()):
        raise ValueError("control population changed")
    out.mkdir(parents=True)
    write_rows(out / "original_workload.jsonl", originals)
    for kind, rs in controls.items():
        write_rows(out / (kind + "_workload.jsonl"), rs)
    write_rows(out / "control_receipts.jsonl", receipts)
    write_json(
        out / "prepare_receipt.json",
        {
            "original_rows": 4188,
            "control_rows": {k: len(rs) for k, rs in controls.items()},
            "unique_control_prompts": {
                k: len({r["prompt_sha256"] for r in rs}) for k, rs in controls.items()
            },
            "max_token_error": max(
                abs(r["actual_tokens"] - r["target_tokens"]) for r in receipts
            ),
            "no_activation_or_outcome_selection": True,
            "workload_sha256": {p.name: file_hash(p) for p in out.glob("*.jsonl")},
        },
    )
    print(
        "judge_signal_prepared",
        4188,
        {k: len(rs) for k, rs in controls.items()},
        flush=True,
    )


if __name__ == "__main__":
    prepare()
