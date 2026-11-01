"""Offline task-grouped decoder fitting and paired held-out uncertainty."""

from __future__ import annotations

import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from experiments.adversarial_crossed_text_probe.common import (
    ROOT,
    bind,
    config,
    digest,
    fold_map,
    load_inputs,
    originals,
    protocol_identity,
    read_jsonl,
    row_metadata,
    selected_rows,
    write_json,
)


def load_captures(folder: Path, split: str, cfg: dict, arm: str) -> tuple[list, dict]:
    variants, _ = load_inputs(cfg)
    identity = {"protocol": protocol_identity(cfg), "arm": arm}
    if json.loads((folder / "identity.json").read_text()) != identity:
        raise ValueError("capture protocol identity mismatch")
    complete = json.loads((folder / f"{split}.complete.json").read_text())
    journal = folder / f"{split}.jsonl"
    if complete["identity"] != identity or complete["journal_sha256"] != digest(
        journal
    ):
        raise ValueError("incomplete or modified capture journal")
    expected = {r["id"]: r for r in selected_rows(variants, split, cfg)}
    rows = read_jsonl(journal)
    if len(rows) != len(expected) or {r["id"] for r in rows} != set(expected):
        raise ValueError("missing/duplicate/unexpected capture rows; no partial report")
    if complete["rows"] != len(rows):
        raise ValueError("completion count mismatch")
    original = originals(variants)
    arrays = defaultdict(list)
    keys = [
        f"L{layer}:{pool}"
        for layer in cfg["arms"][arm]["layers"]
        for pool in cfg["pools"]
    ]
    for row in rows:
        variant = expected[row["id"]]
        m = variant["metadata"]
        expected_meta = row_metadata(variant, original[(str(m["task_id"]), m["code"])])
        if any(row[k] != v for k, v in expected_meta.items()):
            raise ValueError("capture metadata does not match frozen variants")
        if any(
            not np.isfinite(row[k]) for k in ("score", "logit_0", "logit_1", "margin")
        ):
            raise ValueError("nonfinite output")
        path = folder / row["file"]
        if path.parent != folder or digest(path) != row["sha256"]:
            raise ValueError("activation path/checksum mismatch")
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != set(keys):
                raise ValueError("wrong activation keys")
            for key in keys:
                x = data[key]
                if (
                    x.shape != (cfg["arms"][arm]["hidden_size"],)
                    or not np.isfinite(x).all()
                ):
                    raise ValueError("invalid activation vector")
                arrays[key].append(x)
    arrays = {key: np.stack(value) for key, value in arrays.items()}
    arrays["length"] = np.array([r["code_features"] for r in rows])
    return rows, arrays


def fit_linear(x: np.ndarray, y: np.ndarray, c: float) -> dict:
    scaler = StandardScaler().fit(x)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model = LogisticRegression(
            C=c, solver="liblinear", tol=1e-6, max_iter=5000, random_state=0
        ).fit(scaler.transform(x), y)
    return {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
    }


def predict(model: dict, x: np.ndarray) -> np.ndarray:
    return ((x - np.array(model["mean"])) / np.array(model["scale"])) @ np.array(
        model["coef"]
    ) + model["intercept"]


def shuffled_labels(rows: list[dict], seed: int) -> tuple[np.ndarray, dict]:
    tasks = sorted({str(r["task_id"]) for r in rows})
    rng = np.random.default_rng(seed)
    flips = dict(zip(tasks, rng.integers(0, 2, len(tasks)).tolist(), strict=True))
    return np.array([r["label"] ^ flips[str(r["task_id"])] for r in rows]), flips


def select_probe(
    rows: list[dict], arrays: dict, y: np.ndarray, cfg: dict, keys: list[str]
) -> dict:
    if any(r["split"] != "development" or r["condition"] != "none" for r in rows):
        raise ValueError("probe selection accepts only development no-text rows")
    tasks = [str(r["task_id"]) for r in rows]
    folds = fold_map(tasks, cfg)
    assignments = np.array([folds[t] for t in tasks])
    candidates = []
    for key in keys:  # Frozen order defines deterministic tie-breaking.
        for c in cfg["C"]:
            aucs = []
            for fold in range(cfg["folds"]):
                train, val = assignments != fold, assignments == fold
                if len(set(y[train])) != 2 or len(set(y[val])) != 2:
                    raise ValueError("each fold must contain both labels")
                fitted = fit_linear(arrays[key][train], y[train], c)
                aucs.append(
                    float(roc_auc_score(y[val], predict(fitted, arrays[key][val])))
                )
            candidates.append(
                {
                    "key": key,
                    "C": c,
                    "fold_auroc": aucs,
                    "cv_auroc": float(np.mean(aucs)),
                }
            )
    winner = max(candidates, key=lambda r: r["cv_auroc"])
    return {
        "selected": winner,
        "candidates": candidates,
        "folds": folds,
        "linear": fit_linear(arrays[winner["key"]], y, winner["C"]),
    }


def fit_models(rows: list[dict], arrays: dict, cfg: dict, arm: str) -> dict:
    keys = [
        f"L{layer}:{pool}"
        for layer in cfg["arms"][arm]["layers"]
        for pool in cfg["pools"]
    ]
    y = np.array([r["label"] for r in rows])
    shuffled, flips = shuffled_labels(rows, cfg["seed"])
    return {
        "probe": select_probe(rows, arrays, y, cfg, keys),
        "shuffled": select_probe(rows, arrays, shuffled, cfg, keys),
        "length": select_probe(rows, arrays, y, cfg, ["length"]),
        "label_swaps": flips,
    }


def paired_arrays(rows: list[dict], names: list[str]) -> tuple[list[str], dict]:
    tasks = defaultdict(dict)
    for row in rows:
        task, label = str(row["task_id"]), row["label"]
        if label in tasks[task]:
            raise ValueError("duplicate task/label in condition")
        tasks[task][label] = row
    if any(set(pair) != {0, 1} for pair in tasks.values()):
        raise ValueError("incomplete pairs")
    ordered = sorted(tasks)
    return ordered, {
        name: np.array([[tasks[t][label][name] for label in (0, 1)] for t in ordered])
        for name in names
    }


def bootstrap_metrics(
    values: dict[str, np.ndarray],
    resamples: int,
    seed: int,
    differences: list[tuple[str, str]],
    contrasts: dict[str, dict[str, float]] | None = None,
) -> dict:
    """Multinomial counts are task bootstrap weights, including duplicate draws."""
    n = len(next(iter(values.values())))
    if n < 2:
        raise ValueError("need at least two complete tasks for uncertainty")
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n, np.full(n, 1 / n), size=resamples).astype(float)
    result, auc_draws = {}, {}
    for name, scores in values.items():
        if scores.shape != (n, 2) or not np.isfinite(scores).all():
            raise ValueError("invalid paired scores")
        pos, neg = scores[:, 1], scores[:, 0]
        kernel = (pos[:, None] > neg[None, :]).astype(float) + 0.5 * (
            pos[:, None] == neg[None, :]
        )
        draws = np.sum((counts @ kernel) * counts, axis=1) / n**2
        auc_draws[name] = draws
        win = (pos > neg).astype(float) + 0.5 * (pos == neg)
        result[name] = {
            "auroc": float(kernel.mean()),
            "auroc_ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
            "win_rate": float(win.mean()),
            "win_ci95": np.quantile(counts @ win / n, [0.025, 0.975]).tolist(),
        }
    deltas = {}
    for a, b in differences:
        deltas[f"{a}_minus_{b}"] = {
            "difference": result[a]["auroc"] - result[b]["auroc"],
            "ci95": np.quantile(auc_draws[a] - auc_draws[b], [0.025, 0.975]).tolist(),
        }
    for name, weights in (contrasts or {}).items():
        draws = sum(weight * auc_draws[key] for key, weight in weights.items())
        deltas[name] = {
            "difference": sum(
                weight * result[key]["auroc"] for key, weight in weights.items()
            ),
            "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        }
    return {"pairs": n, "metrics": result, "differences": deltas}


def predictions(rows: list[dict], arrays: dict, fit: dict) -> list[dict]:
    outputs = {
        name: predict(fit[name]["linear"], arrays[fit[name]["selected"]["key"]])
        for name in ("probe", "shuffled", "length")
    }
    return [
        {
            **{k: row[k] for k in ("id", "task_id", "label", "condition", "split")},
            **{name: float(values[i]) for name, values in outputs.items()},
            "output": row["score"],
            "output_margin": row["margin"],
        }
        for i, row in enumerate(rows)
    ]


def reports(rows: list[dict], cfg: dict) -> dict:
    result = {}
    names = ["probe", "shuffled", "length", "output", "output_margin"]
    for condition in cfg["conditions"]:
        subset = [r for r in rows if r["condition"] == condition]
        tasks, values = paired_arrays(subset, names)
        result[condition] = {
            **bootstrap_metrics(
                values,
                cfg["bootstrap"],
                cfg["seed"],
                [
                    ("probe", "output"),
                    ("probe", "output_margin"),
                    ("probe", "length"),
                    ("probe", "shuffled"),
                ],
            ),
            "task_ids": tasks,
            "rows": len(subset),
            "missing_computational_rows": 0,
            "descriptive_only": condition == "cover_story",
        }
    return result


def fit_command(folder: Path, cfg: dict, arm: str) -> None:
    rows, arrays = load_captures(folder, "development", cfg, arm)
    # Never refit after held-out collection begins, including partial extraction.
    if (folder / "heldout.fit.json").exists() or (folder / "heldout.jsonl").exists():
        raise ValueError("held-out extraction has begun; fitting is locked")
    import sklearn

    result = {
        "identity": {"protocol": protocol_identity(cfg), "arm": arm},
        "development_complete_sha256": digest(folder / "development.complete.json"),
        "packages": {"numpy": np.__version__, "sklearn": sklearn.__version__},
        **fit_models(rows, arrays, cfg, arm),
    }
    bind(folder / "fit.json", result)


def evaluate_command(folder: Path, cfg: dict, arm: str) -> None:
    from experiments.adversarial_crossed_text_probe.extract import validate_fit_gate

    identity = {"protocol": protocol_identity(cfg), "arm": arm}
    validate_fit_gate(folder, identity)
    if json.loads((folder / "heldout.fit.json").read_text()) != {
        "fit_sha256": digest(folder / "fit.json")
    }:
        raise ValueError("fit changed after held-out extraction began")
    rows, arrays = load_captures(folder, "heldout", cfg, arm)
    fit = json.loads((folder / "fit.json").read_text())
    output = predictions(rows, arrays, fit)
    write_json(
        folder / "predictions.json",
        {"fit_sha256": digest(folder / "fit.json"), "rows": output},
    )
    write_json(
        folder / "report.json",
        {
            "identity": identity,
            "fit_sha256": digest(folder / "fit.json"),
            "conditions": reports(output, cfg),
        },
    )


def compare_command(cfg: dict, arms: list[str]) -> None:
    if len(arms) != 2 or arms[0] == arms[1]:
        raise ValueError("comparison needs two different arms")
    root = ROOT / cfg["output"]
    loaded = []
    for arm in arms:
        folder = root / arm
        pred = json.loads((folder / "predictions.json").read_text())
        if pred["fit_sha256"] != digest(folder / "fit.json"):
            raise ValueError("stale cross-model predictions")
        fit = json.loads((folder / "fit.json").read_text())
        if fit["identity"] != {"protocol": protocol_identity(cfg), "arm": arm}:
            raise ValueError("cross-model protocol mismatch")
        loaded.append({r["id"]: r for r in pred["rows"]})
    a, b = loaded
    if set(a) != set(b):
        raise ValueError(
            "cross-model row coverage differs; report failures before comparison"
        )
    joined = []
    for key in sorted(a):
        if any(
            a[key][k] != b[key][k] for k in ("task_id", "label", "condition", "split")
        ):
            raise ValueError("cross-model row metadata mismatch")
        joined.append(
            {
                **a[key],
                "probe_a": a[key]["probe"],
                "probe_b": b[key]["probe"],
                "output_a": a[key]["output"],
                "output_b": b[key]["output"],
            }
        )
    result = {}
    for condition in cfg["conditions"]:
        tasks, values = paired_arrays(
            [r for r in joined if r["condition"] == condition],
            ["probe_a", "probe_b", "output_a", "output_b"],
        )
        result[condition] = {
            "task_ids": tasks,
            "descriptive_only": condition == "cover_story",
            **bootstrap_metrics(
                values,
                cfg["bootstrap"],
                cfg["seed"],
                [("probe_a", "probe_b"), ("output_a", "output_b")],
                {
                    "probe_output_gap_a_minus_b": {
                        "probe_a": 1,
                        "output_a": -1,
                        "probe_b": -1,
                        "output_b": 1,
                    }
                },
            ),
        }
    write_json(
        root / f"compare-{arms[0]}-{arms[1]}.json", {"arms": arms, "conditions": result}
    )


def main() -> None:
    cfg = config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["fit", "evaluate", "compare"])
    parser.add_argument("--arm", choices=list(cfg["arms"]))
    parser.add_argument("--arms", nargs=2, choices=list(cfg["arms"]))
    args = parser.parse_args()
    if args.command == "compare":
        compare_command(cfg, args.arms or [])
    elif not args.arm:
        parser.error("--arm is required for fit/evaluate")
    else:
        folder = ROOT / cfg["output"] / args.arm
        (fit_command if args.command == "fit" else evaluate_command)(
            folder, cfg, args.arm
        )


if __name__ == "__main__":
    main()
