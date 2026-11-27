"""Offline interface audit for Inkling 2 Mini on the frozen OOD prompts:
rendering, token counts, decision tokens, BOS handling and prompts that carry
literal Inkling control strings. Tokenizer files only; no weights or GPU."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from experiments.evaluation.inkling_ood.interface import (
    MODEL_ID,
    REVISION,
    contains_special_marker,
    render_margin_prompt,
)

ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()
)
INPUT = "data/tool_trajectory_monitoring/teacher_ood_benchmark/prompts.jsonl"
INPUT_SHA256 = "fa4b0d917e51eb91a52efa93cddf0eb154b88993269c97cfdb161c42680136a9"
QWEN_TOTAL, QWEN_MAX = 60_539_619, 30_382
FLINT_TOTAL, FLINT_MAX = 50_935_936, 26_047


def audit(reasoning_effort: str) -> dict:
    from transformers import AutoTokenizer

    data = (ROOT / INPUT).read_bytes()
    if hashlib.sha256(data).hexdigest() != INPUT_SHA256:
        raise ValueError("frozen OOD input checksum differs")
    rows = [json.loads(line) for line in data.splitlines()]
    del data
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    digest = hashlib.sha256()
    lengths: list[int] = []
    marked: dict[str, int] = {}
    with_special = 0
    for start in range(0, len(rows), 64):
        chunk = rows[start : start + 64]
        rendered = [
            render_margin_prompt(
                tokenizer, str(r["prompt"]), reasoning_effort=reasoning_effort
            )
            for r in chunk
        ]
        for row, text in zip(chunk, rendered, strict=True):
            digest.update(text.encode() + b"\0")
            if contains_special_marker(str(row["prompt"])):
                source = row["metadata"]["source_dataset"]
                marked[source] = marked.get(source, 0) + 1
        plain = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        special = tokenizer(rendered, add_special_tokens=True)["input_ids"]
        with_special += sum(
            len(a) != len(b) for a, b in zip(plain, special, strict=True)
        )
        lengths.extend(len(ids) for ids in plain)
    longest = max(range(len(rows)), key=lengths.__getitem__)
    decision = [tokenizer.encode(v, add_special_tokens=False) for v in ("0", "1")]
    tail = tokenizer.convert_ids_to_tokens(
        tokenizer.encode(
            "<|end_message|><|message_model|><|content_text|>Prediction:1",
            add_special_tokens=False,
        )
    )
    return {
        "model": MODEL_ID,
        "revision": REVISION,
        "reasoning_effort": reasoning_effort,
        "rows": len(rows),
        "rendered_sha256": digest.hexdigest(),
        "total_tokens": sum(lengths),
        "max_tokens": max(lengths),
        "longest_id": rows[longest]["id"],
        "ratio_to_qwen_total": sum(lengths) / QWEN_TOTAL,
        "references": {
            "qwen": {"total_tokens": QWEN_TOTAL, "max_tokens": QWEN_MAX},
            "flint": {"total_tokens": FLINT_TOTAL, "max_tokens": FLINT_MAX},
        },
        "decision_token_ids": decision,
        "boundary_tokens": tail,
        "rows_where_special_tokens_change_length": with_special,
        "rows_with_literal_inkling_markers": marked,
    }


if __name__ == "__main__":
    effort = sys.argv[1] if len(sys.argv) > 1 else "none"
    out = ROOT / f"results/inkling_ood/interface_audit_{effort}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    report = audit(effort)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
