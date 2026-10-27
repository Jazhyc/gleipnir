"""Last-token logistic probes over every layer, plus the zero-shot margin.

Standardized L2 logistic regression on the last token's residual stream after
each decoder block. Soft targets are fit by duplicating each row as a positive
and a negative with weights ``p`` and ``1 - p``. Under the prompted backbone,
the base model's own decision margin is reported as the zero-shot reference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gleipnir.evaluation.metrics import evaluate_binary_monitor

ROOT = Path(__file__).resolve().parents[2]
C_GRID = (1e-3, 1e-2, 1e-1, 1.0)


def load(split_dir: Path) -> tuple[np.ndarray, list[dict]]:
    import torch

    bits = torch.from_numpy(np.load(split_dir / "last_layers.npy"))
    rows = [json.loads(line) for line in (split_dir / "rows.jsonl").open()]
    return bits.view(torch.bfloat16).float().numpy(), rows


def macro(rows: list[dict], scores: np.ndarray) -> dict:
    frame = pd.DataFrame(
        {
            "dataset": [r["source"] for r in rows],
            "label": [r["label"] for r in rows],
            "score": scores,
        }
    )
    return evaluate_binary_monitor(frame)["macro"]


def fit(x: np.ndarray, rows: list[dict], objective: str, c: float):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000))
    if objective == "hard":
        return model.fit(x, [r["label"] for r in rows])
    p = np.array([r["soft_target"] for r in rows])
    return model.fit(
        np.concatenate([x, x]),
        np.concatenate([np.ones(len(x)), np.zeros(len(x))]),
        logisticregression__sample_weight=np.concatenate([p, 1 - p]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", required=True)
    parser.add_argument("--population", default="pilot")
    parser.add_argument("--backbone", choices=("prompted", "ntp"), required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    root = Path(args.store_root) / args.population / args.backbone
    train_x, train_rows = load(root / "train")
    id_x, id_rows = load(root / "id")
    result = {"backbone": args.backbone, "probes": []}
    if args.backbone == "prompted":
        margins = np.array([r["margin"] for r in id_rows])
        result["zero_shot_margin"] = macro(id_rows, 1 / (1 + np.exp(-margins)))
    for layer in range(train_x.shape[1]):
        for objective in ("hard", "soft"):
            for c in C_GRID:
                model = fit(train_x[:, layer], train_rows, objective, c)
                scores = model.predict_proba(id_x[:, layer])[:, 1]
                result["probes"].append(
                    {
                        "layer": layer,
                        "objective": objective,
                        "C": c,
                        "id": macro(id_rows, scores),
                    }
                )
        best = max(
            (p for p in result["probes"] if p["layer"] == layer),
            key=lambda p: p["id"]["pauroc_at_20"],
        )
        print(
            f"layer {layer} best {best['objective']} C={best['C']} "
            f"{best['id']['pauroc_at_20']:.4f}",
            flush=True,
        )
    out = ROOT / "results/probe_generalization" / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / f"lasttoken_{args.backbone}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
