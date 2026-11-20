"""Offline interface audit for Flint on the frozen OOD prompts: rendering,
token counts, decision tokens and how many prompts the chat template's
sanitizer rewrites. Tokenizer files only; no weights or GPU."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.flint_ood.interface import (
    MODEL_ID,
    REVISION,
    render_margin_prompt,
    template_changed_user_text,
)

ROOT = Path(__file__).resolve().parents[2]
INPUT = "data/tool_trajectory_monitoring/teacher_ood_benchmark/prompts.jsonl"
INPUT_SHA256 = "fa4b0d917e51eb91a52efa93cddf0eb154b88993269c97cfdb161c42680136a9"
QWEN_TOTAL, QWEN_MAX = 60_539_619, 30_382


def audit() -> dict:
    from transformers import AutoTokenizer

    data = (ROOT / INPUT).read_bytes()
    if hashlib.sha256(data).hexdigest() != INPUT_SHA256:
        raise ValueError("frozen OOD input checksum differs")
    rows = [json.loads(line) for line in data.splitlines()]
    del data
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    digest = hashlib.sha256()
    lengths: list[int] = []
    changed: dict[str, int] = {}
    with_special = 0
    for start in range(0, len(rows), 64):
        chunk = rows[start : start + 64]
        rendered = [render_margin_prompt(tokenizer, str(r["prompt"])) for r in chunk]
        for row, text in zip(chunk, rendered, strict=True):
            digest.update(text.encode() + b"\0")
            if template_changed_user_text(text, str(row["prompt"])):
                source = row["metadata"]["source_dataset"]
                changed[source] = changed.get(source, 0) + 1
        plain = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        special = tokenizer(rendered, add_special_tokens=True)["input_ids"]
        with_special += sum(
            len(a) != len(b) for a, b in zip(plain, special, strict=True)
        )
        lengths.extend(len(ids) for ids in plain)
    longest = max(range(len(rows)), key=lengths.__getitem__)
    decision = [tokenizer.encode(v, add_special_tokens=False) for v in ("0", "1")]
    tail = tokenizer.convert_ids_to_tokens(
        tokenizer.encode("<|assistant|>Prediction:1", add_special_tokens=False)
    )
    return {
        "model": MODEL_ID,
        "revision": REVISION,
        "rows": len(rows),
        "rendered_sha256": digest.hexdigest(),
        "total_tokens": sum(lengths),
        "max_tokens": max(lengths),
        "longest_id": rows[longest]["id"],
        "ratio_to_qwen_total": sum(lengths) / QWEN_TOTAL,
        "qwen_reference": {"total_tokens": QWEN_TOTAL, "max_tokens": QWEN_MAX},
        "decision_token_ids": decision,
        "boundary_tokens": tail,
        "rows_where_special_tokens_change_length": with_special,
        "rows_rewritten_by_template_sanitizer": changed,
    }


if __name__ == "__main__":
    out = ROOT / "results/flint_ood/interface_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    report = audit()
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
