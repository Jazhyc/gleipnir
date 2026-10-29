"""Render and tokenize the 27B probe workloads (on the Pod, with the 27B tokenizer).

Each backbone's rows are rendered and tokenized exactly as in
``probe_generalization/extract.py``. Rows: ``{id, source, label, input, tokens,
keep_from, key}`` for the ID and OOD splits and the 64 diagnostic OOD rows, in
``data/probe_optimized_serving/qwen38_27b/<backbone>_<split>.jsonl``.
``keep_from`` is the first kept token (NTP: 1, dropping the attention sink;
prompted: the trajectory's first token), and ``keep_prompted.json`` maps each
prompted row's token key to it for the probe pooler.
"""

from __future__ import annotations

from pathlib import Path

from experiments.probe_generalization.extract import MODELS, render, tokenize
from experiments.probe_optimized_serving.keys import token_key
from gleipnir.data.monitoring import read_rows, write_json, write_rows

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "id": ROOT / "data/probe_generalization/full/id.jsonl",
    "ood": ROOT / "data/probe_generalization/full/ood.jsonl",
    "diag": ROOT / "data/probe_generalization/optimized_diag/ood.jsonl",
}
OUT = ROOT / "data/probe_optimized_serving/qwen38_27b"


def main() -> None:
    from transformers import AutoTokenizer

    model_id, revision, _, _ = MODELS["qwen38_27b"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    OUT.mkdir(parents=True, exist_ok=True)
    for backbone in ("ntp", "prompted"):
        keep = {}
        for split, source in SOURCES.items():
            rows = []
            for row in read_rows(source):
                text, keep_char = render(tokenizer, row, backbone)
                ids, keep_from = tokenize(tokenizer, text, keep_char)
                if backbone == "ntp":
                    keep_from = 1  # the training store's sink skip
                key = token_key(ids)
                if keep.setdefault(key, keep_from) != keep_from:
                    raise ValueError(f"conflicting kept starts for {row['id']}")
                rows.append(
                    {
                        "id": row["id"],
                        "source": row["source"],
                        "label": row["label"],
                        "input": text,
                        "tokens": len(ids),
                        "keep_from": keep_from,
                        "key": key,
                    }
                )
            write_rows(OUT / f"{backbone}_{split}.jsonl", rows)
            print(backbone, split, len(rows), sum(r["tokens"] for r in rows))
        if backbone == "prompted":
            write_json(OUT / "keep_prompted.json", keep)


if __name__ == "__main__":
    main()
