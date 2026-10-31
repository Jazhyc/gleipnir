"""CPU-only input contracts, task splits, token spans and artifact integrity."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from gleipnir.adversarial.apps import SUBMIT, parse
from gleipnir.adversarial.contract import CLOSE, render

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def config() -> dict:
    return json.loads((HERE / "config.json").read_text())


def read_jsonl(path: Path) -> list[dict]:
    return (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def append(path: Path, value: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(value, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def check_deadline(path: Path, now: float | None = None) -> dict:
    """Same immutable deadline guard as crossed_text/campaign.py, fail closed."""
    if not path.exists():
        raise SystemExit("VM campaign deadline must be recorded before GPU work")
    record = json.loads(path.read_text())
    deadline = record.get("deadline_epoch")
    if not isinstance(deadline, (int, float)) or not math.isfinite(deadline):
        raise SystemExit("invalid campaign VM deadline")
    if (time.time() if now is None else now) >= deadline:
        raise SystemExit("campaign VM deadline reached")
    return record


def protocol_identity(cfg: dict) -> dict:
    return {
        "config": cfg,
        "sources": {
            p.name: digest(p)
            for p in sorted(HERE.iterdir())
            if p.suffix in {".py", ".json", ".md"}
        },
    }


def require_committed() -> None:
    paths = [str(HERE.relative_to(ROOT))]
    tracked = subprocess.check_output(
        ["git", "ls-files", "--", *paths], cwd=ROOT, text=True
    )
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *paths], cwd=ROOT, text=True
    )
    if not tracked.strip() or dirty.strip():
        raise SystemExit("commit the frozen probe protocol and code before extraction")


def bind(path: Path, value: dict) -> None:
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f"artifact identity changed: {path}")
    write_json(path, value)


def validate_rows(rows: list[dict], selection: dict) -> None:
    splits = {s: set(map(str, selection[s])) for s in ("development", "heldout")}
    if splits["development"] & splits["heldout"]:
        raise ValueError("development/heldout task overlap")
    ids, cells = set(), defaultdict(list)
    for row in rows:
        m = row["metadata"]
        if row["id"] in ids:
            raise ValueError("duplicate row id")
        ids.add(row["id"])
        if m["split"] not in splits:
            continue
        if str(m["task_id"]) not in splits[m["split"]]:
            raise ValueError("row task disagrees with selection")
        if m["label"] != {"honest": 0, "backdoored": 1}[m["code"]]:
            raise ValueError("code label mismatch")
        cells[(m["split"], str(m["task_id"]), m["condition"])].append(m)
    for values in cells.values():
        if sorted(v["label"] for v in values) != [0, 1]:
            raise ValueError("incomplete or duplicate code pair")
        if len({v["rendered_header_sha256"] for v in values}) != 1:
            raise ValueError("crossed texts differ")


def load_inputs(cfg: dict) -> tuple[list[dict], dict]:
    for name, expected in cfg["input_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"pinned input changed: {name}")
    folder = ROOT / cfg["input"]
    manifest = json.loads((folder / "variants.manifest.json").read_text())
    if manifest["dry_run"]:
        raise ValueError("mock Step 1 inputs refused")
    for name, expected in manifest["sources"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"sealed Step 1 source changed: {name}")
    rows = read_jsonl(folder / "variants.jsonl")
    selection = json.loads((folder / "selection.json").read_text())
    validate_rows(rows, selection)
    return rows, selection


def selected_rows(rows: list[dict], split: str, cfg: dict) -> list[dict]:
    return [
        r
        for r in rows
        if r["metadata"]["split"] == split
        and r["metadata"]["condition"]
        in (["none"] if split == "development" else cfg["conditions"])
    ]


def fold_map(tasks: list[str], cfg: dict) -> dict[str, int]:
    ordered = sorted(set(tasks), key=lambda t: (sha(cfg["fold_prefix"] + t), t))
    return {t: i % cfg["folds"] for i, t in enumerate(ordered)}


def originals(rows: list[dict]) -> dict[tuple[str, str], str]:
    return {
        (str(r["metadata"]["task_id"]), r["metadata"]["code"]): parse(r["prompt"]).code
        for r in rows
        if r["metadata"].get("condition") == "none"
    }


def code_spans(request: str, original: str, added: int) -> list[tuple[int, int]]:
    """Character spans of original program, excluding the inserted header."""
    edited = parse(request).code
    start = request.index(SUBMIT) + len(SUBMIT)
    if not added:
        if edited != original:
            raise ValueError("no-text code changed")
        return [(start, start + len(edited))]
    # Exact Step 1 insertion rule, without importing its generation machinery.
    import re

    at = re.match(
        r"^(#!.*\n)?([ \t\f]*#.*coding[:=][ \t]*[-\w.]+.*\n)?", original
    ).end()
    if edited[:at] + edited[at + added :] != original:
        raise ValueError("removing header does not recover original code")
    return [(start, start + at), (start + at + added, start + len(edited))]


def code_token_indices(
    tokenizer: Any, request: str, original: str, added: int, ids: list[int]
) -> list[int]:
    rendered = render(tokenizer, request)
    # The Qwen chat templates trim the user message, dropping CLOSE's final
    # newline, so align on the right-trimmed request (checked on both
    # tokenizers, 24 October). Code spans end well before that newline.
    body = request.rstrip()
    if rendered.count(body) != 1 or not request.endswith(CLOSE):
        raise ValueError("ambiguous rendered prompt alignment")
    offset = rendered.index(body)
    spans = [(a + offset, b + offset) for a, b in code_spans(request, original, added)]
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    if list(encoded["input_ids"]) != ids:
        raise ValueError("offset tokenization differs from Scorer.encode")
    indices = [
        i
        for i, (a, b) in enumerate(encoded["offset_mapping"])
        if a < b and any(lo <= a and b <= hi for lo, hi in spans)
    ]
    if not indices or len(indices) >= len(ids) or len(ids) - 1 in indices:
        raise ValueError("empty/invalid code token span")
    return indices


def row_metadata(row: dict, original: str) -> dict:
    m = row["metadata"]
    return {
        "id": row["id"],
        **{k: m[k] for k in ("task_id", "code", "label", "condition", "split")},
        "prompt_sha256": sha(row["prompt"]),
        "code_features": [
            math.log1p(len(original.encode())),
            math.log1p(len(original.splitlines())),
        ],
    }
