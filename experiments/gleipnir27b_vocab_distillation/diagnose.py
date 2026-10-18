"""Step 1 diagnostic: is there label information beyond the 0/1 margin?

Implements the design frozen in this directory's README from ``extract.py``
outputs: parity with the reference scorers, ID quality per model x prompt,
probability outside {0, 1}, train->ID logistic probes (margin; margin plus
tail log-probabilities; full hidden state), and base vs Gleipnir changes.
Run on a GPU host (tail features need each model's LM head).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.gleipnir27b_vocab_distillation.extract import MODELS, OUT, ROOT
from experiments.qwen27b_generations_ood.compare_id_train import macro

TAIL_TOKENS = 256
C_GRID = (0.001, 0.01, 0.1, 1.0)


def load(name: str, job: str) -> tuple[list[dict], np.ndarray]:
    path = OUT / name / job
    rows = [json.loads(line) for line in (path / "rows.jsonl").read_text().splitlines()]
    return rows, np.load(path / "hidden.npy")


def sigmoid(x):
    return 1 / (1 + np.exp(-np.asarray(x, dtype=np.float64)))


def quality(rows: list[dict], scores) -> dict:
    return macro(
        [(r["source"], r["label"], float(s)) for r, s in zip(rows, scores, strict=True)]
    )


def parity(rows: list[dict], path: Path, key: str) -> dict:
    reference = {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        reference[str(record["id"])] = float(record[key])
    diff = np.array([r["margin"] - reference[r["id"]] for r in rows])
    ours = np.array([r["margin"] for r in rows])
    theirs = np.array([reference[r["id"]] for r in rows])
    return {
        "rows": len(rows),
        "margin_mae": float(np.abs(diff).mean()),
        "margin_max": float(np.abs(diff).max()),
        "correlation": float(np.corrcoef(ours, theirs)[0, 1]),
        "threshold_flips": int(((ours >= 0) != (theirs >= 0)).sum()),
    }


def head_features(model: str, hidden: np.ndarray, tokens=None):
    """FP32 log-probs of selected tokens, plus BF16-head margins for tie counts."""
    import torch

    from experiments.gleipnir27b_vocab_distillation.extract import lm_head

    weight, _ = lm_head(Path(MODELS[model]))
    out_logprobs, bf16_margin, mass = [], [], []
    for start in range(0, len(hidden), 256):
        h = torch.tensor(hidden[start : start + 256], device="cuda")
        logits = h @ weight.float().T
        logprobs = torch.log_softmax(logits, dim=-1)
        if tokens is None:
            mass.append(logprobs.exp().sum(0).cpu().numpy())
        else:
            out_logprobs.append(logprobs[:, tokens].cpu().numpy())
        bf16 = (h.bfloat16() @ weight[15:17].T).float()
        bf16_margin.append((bf16[:, 1] - bf16[:, 0]).cpu().numpy())
    del weight
    torch.cuda.empty_cache()
    if tokens is None:
        return np.sum(mass, axis=0), np.concatenate(bf16_margin)
    return np.concatenate(out_logprobs), np.concatenate(bf16_margin)


def probe(train_x, train_y, groups) -> tuple[object, dict]:
    """Standardized L2 logistic regression; C chosen by grouped 5-fold CV."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def model(c):
        return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000))

    cv = {}
    for c in C_GRID:
        scores = np.zeros(len(train_y))
        for fit, held in GroupKFold(5).split(train_x, train_y, groups):
            scores[held] = (
                model(c)
                .fit(train_x[fit], train_y[fit])
                .predict_proba(train_x[held])[:, 1]
            )
        cv[c] = float(roc_auc_score(train_y, scores))
    best = max(cv, key=cv.get)
    return model(best).fit(train_x, train_y), {"c": best, "cv_auroc": cv}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="extract01")
    args = parser.parse_args()
    report: dict = {"name": args.name}
    lineage = {
        (r := json.loads(line))["index"]: r["lineage_group"]
        for line in (ROOT / "data/qwen38-27b-s0-01/strict_filter/training.jsonl")
        .read_text()
        .splitlines()
    }
    kimi = {
        (r := json.loads(line))["index"]: r["soft_target"]
        for line in (ROOT / "data/qwen38-27b-s0-01/strict_filter/teacher.jsonl")
        .read_text()
        .splitlines()
    }
    data = {}
    for model in ("gleipnir", "base"):
        for job in ("student_id", "teacher_id", "teacher_train"):
            data[(model, job)] = load(args.name, f"{model}_{job}")
    if (OUT / args.name / "gleipnir_teacher_ood").is_dir():
        data[("gleipnir", "teacher_ood")] = load(args.name, "gleipnir_teacher_ood")

    # 1. Parity and quality.
    report["parity"] = {
        "gleipnir_student_id_vs_campaign": parity(
            data[("gleipnir", "student_id")][0],
            ROOT / "results/qwen38-27b-s0-01/evaluation/id.jsonl",
            "margin",
        ),
        "base_teacher_id_vs_prior": parity(
            data[("base", "teacher_id")][0],
            ROOT / "results/qwen27b_generations_ood/qwen38_27b_id/predictions.jsonl",
            "logit_margin_1_minus_0",
        ),
        "base_teacher_train_vs_prior": parity(
            data[("base", "teacher_train")][0],
            ROOT / "results/qwen27b_generations_ood/qwen38_27b_train/predictions.jsonl",
            "logit_margin_1_minus_0",
        ),
    }
    report["quality"] = {
        f"{model}_{job}": quality(rows, sigmoid([r["margin"] for r in rows]))
        for (model, job), (rows, _) in data.items()
        if not job.endswith("_train")
    }

    # 2. Mass outside {0, 1}.
    mass = {}
    for (model, job), (rows, _) in data.items():
        by = defaultdict(list)
        for r in rows:
            by[f"{r['source']}/{r['label']}"].append(r["outside_01_mass"])
        values = [r["outside_01_mass"] for r in rows]
        mass[f"{model}_{job}"] = {
            "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.9)),
            "max": float(np.max(values)),
            "by_source_label": {k: float(np.median(v)) for k, v in sorted(by.items())},
        }
    report["outside_01_mass"] = mass

    # 3. Probes: train on the 7,836 student rows, evaluate on ID (teacher prompt)
    # and, for Gleipnir, on the OOD suite (teacher prompt) when it was extracted.
    report["probes"] = {}
    for model in ("gleipnir", "base"):
        train_rows, train_hidden = data[(model, "teacher_train")]
        keep = [i for i, r in enumerate(train_rows) if r["student_training_row"]]
        train_rows = [train_rows[i] for i in keep]
        train_hidden = train_hidden[keep]
        test_rows, test_hidden = data[(model, "teacher_id")]
        total, train_bf16 = head_features(model, train_hidden)
        total[[15, 16]] = -np.inf
        total[248077:] = -np.inf  # padding rows
        tokens = np.argsort(-total)[:TAIL_TOKENS]
        train_tail, _ = head_features(model, train_hidden, tokens)
        test_tail, test_bf16 = head_features(model, test_hidden, tokens)
        y = np.array([r["label"] for r in train_rows])
        groups = np.array([lineage[r["id"]] for r in train_rows])
        margin = np.array([[r["margin"]] for r in train_rows])
        test_margin = np.array([[r["margin"]] for r in test_rows])

        def centered(x):
            return x - x.mean(axis=1, keepdims=True)

        arms = {
            "margin": (margin, test_margin),
            "margin_plus_tail": (
                np.hstack([margin, centered(train_tail)]),
                np.hstack([test_margin, centered(test_tail)]),
            ),
            "hidden": (train_hidden, test_hidden),
        }
        ood = data.get((model, "teacher_ood"))
        if ood is not None:
            ood_rows, ood_hidden = ood
            ood_tail, _ = head_features(model, ood_hidden, tokens)
            ood_margin = np.array([[r["margin"]] for r in ood_rows])
            ood_x = {
                "margin": ood_margin,
                "margin_plus_tail": np.hstack([ood_margin, centered(ood_tail)]),
                "hidden": ood_hidden,
            }
        results = {}
        for arm, (train_x, test_x) in arms.items():
            fitted, fit = probe(train_x, y, groups)
            results[arm] = {
                **fit,
                "id": quality(test_rows, fitted.predict_proba(test_x)[:, 1]),
            }
            if ood is not None:
                results[arm]["ood"] = quality(
                    ood_rows, fitted.predict_proba(ood_x[arm])[:, 1]
                )
        ties = {
            "fp32_unique_margins": len({round(r["margin"], 6) for r in test_rows}),
            "bf16_unique_margins": len(set(np.round(test_bf16, 6).tolist())),
            "rows": len(test_rows),
        }
        report["probes"][model] = {
            "tail_tokens": tokens[:32].tolist(),
            "tail_train_mass_share": float(
                total[tokens].sum() / total[np.isfinite(total)].sum()
            ),
            "arms": results,
            "id_margin_ties": ties,
            "train_margin_vs_kimi_logit_corr": float(
                np.corrcoef(
                    margin[:, 0],
                    [np.log(kimi[r["id"]] / (1 - kimi[r["id"]])) for r in train_rows],
                )[0, 1]
            ),
            "train_bf16_vs_fp32_margin_mae": float(
                np.abs(train_bf16 - margin[:, 0]).mean()
            ),
        }

    # 4. What training changed (same rows, same prompt).
    changed = {}
    for job in ("teacher_train", "teacher_id"):
        g = {r["id"]: r for r in data[("gleipnir", job)][0]}
        b = {r["id"]: r for r in data[("base", job)][0]}
        ids = sorted(g)
        gm = np.array([g[i]["margin"] for i in ids])
        bm = np.array([b[i]["margin"] for i in ids])
        changed[job] = {
            "margin_correlation": float(np.corrcoef(gm, bm)[0, 1]),
            "median_abs_margin_gleipnir": float(np.median(np.abs(gm))),
            "median_abs_margin_base": float(np.median(np.abs(bm))),
            "top8_overlap": float(
                np.mean(
                    [
                        len(set(g[i]["top_ids"][:8]) & set(b[i]["top_ids"][:8])) / 8
                        for i in ids
                    ]
                )
            ),
        }
    report["training_change"] = changed
    out = OUT / args.name / "diagnostic.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("parity", "probes")}, indent=1)[:6000])


if __name__ == "__main__":
    main()
