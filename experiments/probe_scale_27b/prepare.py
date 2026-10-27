"""Materialize the scale-pilot rows (CPU, stdlib only).

``train`` and ``id`` are byte-identical copies of the frozen probe-study pilot
splits (1,000 rows each), so 27B probes train and select on exactly the rows
the 4B pilot probes used. ``ood`` is a 1,000-row sample of the frozen 6,395-row
OOD suite, stratified by source and label with the probe study's seeded
largest-remainder rule.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

from experiments.probe_generalization.prepare import stratified
from gleipnir.data.monitoring import file_hash, read_rows, write_json, write_rows

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/probe_generalization"
OUT = SOURCE / "scale_pilot"
OOD_ROWS = 1000


def describe(path: Path, rows: list[dict]) -> dict:
    return {
        "rows": len(rows),
        "sha256": file_hash(path),
        "ids_sha256": hashlib.sha256(
            "\n".join(r["id"] for r in rows).encode()
        ).hexdigest(),
        "trajectory_chars": sum(len(r["trajectory"]) for r in rows),
        "by_source_label": dict(
            sorted(Counter(f"{r['source']}/{r['label']}" for r in rows).items())
        ),
    }


def main() -> None:
    pilot = json.loads((SOURCE / "pilot/manifest.json").read_text())
    full = json.loads((SOURCE / "full/manifest.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    splits = {}
    for split in ("train", "id"):
        source = SOURCE / "pilot" / f"{split}.jsonl"
        if file_hash(source) != pilot["splits"][split]["sha256"]:
            raise ValueError(f"pilot {split} changed since its manifest")
        shutil.copyfile(source, OUT / f"{split}.jsonl")
        splits[split] = describe(OUT / f"{split}.jsonl", read_rows(source))
    source = SOURCE / "full/ood.jsonl"
    if file_hash(source) != full["splits"]["ood"]["sha256"]:
        raise ValueError("full OOD rows changed since their manifest")
    sample = stratified(read_rows(source), OOD_ROWS, ("source", "label"))
    write_rows(OUT / "ood.jsonl", sample)
    splits["ood"] = describe(OUT / "ood.jsonl", sample)
    write_json(
        OUT / "manifest.json",
        {
            "population": "scale_pilot",
            "inputs": {
                "pilot_manifest_sha256": file_hash(SOURCE / "pilot/manifest.json"),
                "full_manifest_sha256": file_hash(SOURCE / "full/manifest.json"),
            },
            "ood_rule": "stratified by (source, label), seed 0, largest remainder",
            "splits": splits,
        },
    )
    print(json.dumps(splits, indent=2))


if __name__ == "__main__":
    main()
