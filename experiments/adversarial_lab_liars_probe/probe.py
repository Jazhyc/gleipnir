"""Row join, seed folds and nested cross-validated linear probes.

Torch-free. Activations are the lab-liars decision-token residuals, one
`(layers, hidden)` array per row, stacked in the monitor's configured layer
order (`experiments/adversarial_lab_liars/run.py`, `Capture.array`).
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from experiments.adversarial_lab_liars.common import read_jsonl
from gleipnir.adversarial.eval_sets import ROOT
from gleipnir.adversarial.metrics import threshold_at_fpr

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text())
DATA = ROOT / CONFIG["data"]
INPUT = ROOT / CONFIG["input"]
OUTPUT = ROOT / CONFIG["output"]
POST_CALL = re.compile(r"^board_post\(", re.M)


def count_posts(prompt: str) -> int:
    """Own `board_post` calls in the rendered action-only trajectory."""
    return len(POST_CALL.findall(prompt))


def join_rows(
    meta_rows: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """One row per scored id: metadata, monitor score, activation file, posts.

    The latest `ok` prediction per id wins (as in lab-liars `analyze.py`).
    Every metadata row within the token limit must have exactly one ok score,
    and no prediction may lack metadata.
    """
    meta = {}
    for r in meta_rows:
        if r["id"] in meta:
            raise ValueError(f"duplicate metadata id {r['id']}")
        meta[r["id"]] = r
    latest: dict[str, dict] = {}
    for p in predictions:
        if p["id"] not in meta:
            raise ValueError(f"prediction without metadata: {p['id']}")
        if p["id"] not in latest or p.get("status") == "ok":
            latest[p["id"]] = p
    out = []
    for i, r in meta.items():
        over = r["metadata"]["over_limit"]
        p = latest.get(i)
        if over:
            if p is not None and p.get("status") == "ok":
                raise ValueError(f"over-limit row was scored: {i}")
            continue
        if p is None or p.get("status") != "ok":
            raise ValueError(f"row within the limit has no ok score: {i}")
        if p["seed"] != r["metadata"]["seed"] or p["worker"] != r["metadata"]["worker"]:
            raise ValueError(f"prediction and metadata disagree: {i}")
        out.append(
            {
                **r["metadata"],
                "id": i,
                "score": p["score"],
                "activation_file": p["activation_file"],
                "posts": count_posts(r["prompt"]),
            }
        )
    return out


def load_monitor(monitor: str) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Joined rows plus feature arrays keyed `L<layer>` and `length`."""
    spec = CONFIG["monitors"][monitor]
    folder = INPUT / monitor / "main"
    identity = json.loads((folder / "identity.json").read_text())
    if identity["layers"] != spec["layers"]:
        raise ValueError("activation layer order differs from config")
    rows = join_rows(
        read_jsonl(DATA / spec["metadata"]), read_jsonl(folder / "predictions.jsonl")
    )
    stack = []
    for r in rows:
        a = np.load(folder / "activations" / r["activation_file"])
        if a.shape != (len(spec["layers"]), spec["hidden_size"]):
            raise ValueError(f"activation shape {a.shape} for {r['id']}")
        if not np.isfinite(a).all():
            raise ValueError(f"nonfinite activation for {r['id']}")
        stack.append(a)
    acts = np.stack(stack).astype(np.float64)
    arrays = {f"L{layer}": acts[:, k] for k, layer in enumerate(spec["layers"])}
    arrays["length"] = np.array(
        [[r[spec["token_field"]], r["posts"]] for r in rows], dtype=np.float64
    )
    return rows, arrays


def seed_folds(seeds: list[int]) -> dict[int, int]:
    """Leave-one-seed-out: fold k holds every row of the k-th sorted seed."""
    return {s: k for k, s in enumerate(sorted(set(seeds)))}


def fit_linear(x: np.ndarray, y: np.ndarray, c: float) -> dict[str, Any]:
    s = CONFIG["solver"]
    scaler = StandardScaler().fit(x)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model = LogisticRegression(
            C=c,
            solver=s["name"],
            tol=s["tol"],
            max_iter=s["max_iter"],
            random_state=s["random_state"],
        ).fit(scaler.transform(x), y)
    return {"scaler": scaler, "model": model}


def predict(fitted: dict[str, Any], x: np.ndarray) -> np.ndarray:
    """Decision function (log-odds); monotone in the probe's probability."""
    return fitted["model"].decision_function(fitted["scaler"].transform(x))


def select(
    arrays: dict[str, np.ndarray],
    y: np.ndarray,
    seeds: np.ndarray,
    keys: list[str],
    cs: list[float],
) -> dict[str, Any]:
    """Inner leave-one-seed-out on training rows only; mean per-fold AUROC.

    Folds without both classes are skipped. Ties go to the first candidate in
    the frozen order (keys, then C ascending).
    """
    candidates = []
    for key in keys:
        for c in cs:
            aucs = []
            for s in sorted(set(seeds.tolist())):
                tr, va = seeds != s, seeds == s
                if len(set(y[tr])) != 2 or len(set(y[va])) != 2:
                    continue
                fitted = fit_linear(arrays[key][tr], y[tr], c)
                aucs.append(
                    float(roc_auc_score(y[va], predict(fitted, arrays[key][va])))
                )
            candidates.append({"key": key, "C": c, "cv_auroc": float(np.mean(aucs))})
    best = max(range(len(candidates)), key=lambda i: (candidates[i]["cv_auroc"], -i))
    return {**candidates[best], "candidates": candidates}


def _outer_fold(arrays, y, seeds, train, score, held, keys, cs):
    tr = train & (seeds != held)
    chosen = select({k: arrays[k][tr] for k in keys}, y[tr], seeds[tr], keys, cs)
    fitted = fit_linear(arrays[chosen["key"]][tr], y[tr], chosen["C"])
    te = score & (seeds == held)
    return (
        held,
        np.flatnonzero(te),
        predict(fitted, arrays[chosen["key"]][te]),
        {
            "seed": int(held),
            "key": chosen["key"],
            "C": chosen["C"],
            "inner_cv_auroc": chosen["cv_auroc"],
            "train_rows": int(tr.sum()),
        },
    )


def nested_oof(
    arrays: dict[str, np.ndarray],
    y: np.ndarray,
    seeds: np.ndarray,
    train: np.ndarray,
    score: np.ndarray,
    keys: list[str],
    cs: list[float] | None = None,
    workers: int | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Out-of-fold scores for `score` rows; NaN elsewhere.

    For each held-out seed, layer and C are chosen by inner CV on the training
    rows of the other seeds, the probe is refit on all of them, and it scores
    the held-out seed's `score` rows. A row is never scored by a probe whose
    training rows share its seed.
    """
    cs = cs or CONFIG["C"]
    out = np.full(len(y), np.nan)
    jobs = sorted(set(seeds[score].tolist()))
    results = Parallel(n_jobs=workers or CONFIG["workers"])(
        delayed(_outer_fold)(arrays, y, seeds, train, score, s, keys, cs) for s in jobs
    )
    selections = []
    for _held, index, values, sel in results:
        out[index] = values
        selections.append(sel)
    return out, selections


def shuffle_within_seed(
    y: np.ndarray, seeds: np.ndarray, mask: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Permute labels among `mask` rows within each seed; others unchanged."""
    out = y.copy()
    for s in sorted(set(seeds[mask].tolist())):
        idx = np.flatnonzero(mask & (seeds == s))
        out[idx] = y[rng.permutation(idx)]
    return out


def calibration_mask(rows: list[dict[str, Any]]) -> np.ndarray:
    """Ordinary workers (A, D, E, F) in honest episodes: no lie was told.

    The same 208 rows as lab-liars line (b) (`analyze.line_b`).
    """
    return np.array([r["role"] != "recruiter" and r["honest_episode"] for r in rows])


def line_at(scores: np.ndarray, fpr: float) -> float:
    """1% line: flag `score > line` on at most `fpr` of the calibration rows."""
    if np.isnan(scores).any():
        raise ValueError("calibration rows need a score")
    return threshold_at_fpr(scores, fpr)
