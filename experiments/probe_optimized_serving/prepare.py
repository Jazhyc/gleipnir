"""Draw the 64-row diagnostic OOD sample (CPU, stdlib only).

The rows are stratified by source and label with the probe study's seeded
largest-remainder rule. They get a BF16 Transformers extraction on the Pod
(``probe_generalization/extract.py --population optimized_diag``) for the
activation-level comparison with the optimized capture.
"""

from __future__ import annotations

from pathlib import Path

from experiments.probe_generalization.prepare import stratified
from gleipnir.data.monitoring import file_hash, read_rows, write_json, write_rows

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/probe_generalization/full/ood.jsonl"
OUT = ROOT / "data/probe_generalization/optimized_diag"
ROWS = 64


def main() -> None:
    sample = stratified(read_rows(SOURCE), ROWS, ("source", "label"))
    OUT.mkdir(parents=True, exist_ok=True)
    write_rows(OUT / "ood.jsonl", sample)
    (OUT / "ids.txt").write_text("".join(r["id"] + "\n" for r in sample))
    write_json(
        OUT / "manifest.json",
        {
            "source_sha256": file_hash(SOURCE),
            "rows": len(sample),
            "rule": "stratified by (source, label), seed 0, largest remainder",
            "ood_sha256": file_hash(OUT / "ood.jsonl"),
        },
    )


if __name__ == "__main__":
    main()
