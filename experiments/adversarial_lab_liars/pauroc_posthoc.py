"""Post-hoc pAUROC@20 for the primary liar-vs-honest-recruiter comparison.

Requested 27 Oct; not pre-registered. Same rows and seed bootstrap as
analyze.py's primary AUROC, with the codebase's normalized_partial_auroc.
Writes results/adversarial_lab_liars/pauroc20_posthoc.json.

    python experiments/adversarial_lab_liars/pauroc_posthoc.py
"""

from __future__ import annotations

import json

import numpy as np

from experiments.adversarial_lab_liars.analyze import (
    HONEST,
    LIAR,
    MONITORS,
    group,
    scored,
)
from experiments.adversarial_lab_liars.common import CONFIG, OUTPUT, seed_bootstrap
from gleipnir.evaluation.metrics import normalized_partial_auroc


def pauroc(rows: list[dict], pos_ids: set[str]) -> float | None:
    y = np.array([r["id"] in pos_ids for r in rows], dtype=int)
    if len(set(y)) < 2:
        return None
    return normalized_partial_auroc(y, np.array([r["score"] for r in rows]))


def main() -> None:
    cells = set(CONFIG["analysis"]["primary_cells"])
    out = {}
    for monitor in (*MONITORS, "qwen_base"):
        if not (OUTPUT / monitor / "main" / "predictions.jsonl").exists():
            continue
        main_rows = scored(monitor)["main"]
        pre = [
            r
            for r in main_rows
            if r["role"] == "recruiter"
            and r["analysis_recruiter"]
            and r["cell"] in cells
        ]
        pos, neg = group(pre, LIAR), group(pre, HONEST)
        ids = {r["id"] for r in pos}
        rows = pos + neg
        out[monitor] = {
            "positives": len(pos),
            "negatives": len(neg),
            "pauroc20": pauroc(rows, ids),
            **seed_bootstrap(rows, lambda d, ids=ids: pauroc(d, ids)),
        }
        print(monitor, json.dumps(out[monitor]))
    (OUTPUT / "pauroc20_posthoc.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
