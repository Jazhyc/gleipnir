"""Task-disjoint exploratory APPS fusion of existing black-box SDPA scores."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from gleipnir.data.monitoring import file_hash, read_rows, write_json, write_rows
from gleipnir.evaluation.injection_presence import TEMPLATES, diagnostics
from gleipnir.evaluation.score_fusion import (
    combined_scores,
    fit_combiner,
    fit_threshold,
)
from gleipnir.serving.monitor_score import validate_score_response

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ARMS = ("safety_raw", "safety_calibrated", "two_score", "interaction")


def join_scores(safety: list[dict], presence: list[dict]) -> list[dict]:
    """Require exact source identity despite different question envelopes."""
    lookup = {r["id"]: r for r in presence}
    if len(lookup) != len(presence) or len({r["id"] for r in safety}) != len(safety):
        raise ValueError("duplicate prediction identities")
    if set(lookup) != {r["id"] for r in safety}:
        raise ValueError("prediction population mismatch")
    output = []
    for row in safety:
        injection = lookup[row["id"]]
        for key in ("condition", "task_id", "sample_sha256"):
            if row[key] != injection["original_metadata"][key]:
                raise ValueError(f"source identity mismatch: {key}")
        if row["ground_truth"] != injection["behavior_label"]:
            raise ValueError("behavior label mismatch")
        for scored in (row, injection):
            validate_score_response(scored, scored["prompt_tokens"])
        output.append(
            {
                "id": row["id"],
                "task_id": row["task_id"],
                "condition": row["condition"],
                "sample_sha256": row["sample_sha256"],
                "ground_truth": row["ground_truth"],
                "safety_margin": row["margin"],
                "injection_margin": injection["margin"],
                "safety_score": row["score"],
                "injection_score": injection["score"],
                "safety_prompt_sha256": row["prompt_sha256"],
                "injection_prompt_sha256": injection["prompt_sha256"],
            }
        )
    return output


def report(rows: list[dict], threshold: float) -> dict:
    """Report behavior discrimination inside injection and control strata."""

    def measure(selected: list[dict]) -> dict:
        value = diagnostics(selected, threshold)
        value["thresholds"]["fit_negatives_2pct"] = value["thresholds"].pop(
            "clean_2pct"
        )
        return value

    matched = {}
    for prefix in ("inj", "neutral"):
        for template in TEMPLATES:
            selected = [r for r in rows if r["condition"] == f"{prefix}_{template}"]
            paired = defaultdict(dict)
            for row in selected:
                paired[row["task_id"]][row["ground_truth"]] = row
            both = [
                r
                for pair in paired.values()
                if set(pair) == {0, 1}
                for r in pair.values()
            ]
            matched[f"{prefix}_{template}"] = measure(both)
    return {
        "pooled": measure(rows),
        "by_condition": {
            c: measure([r for r in rows if r["condition"] == c])
            for c in sorted({r["condition"] for r in rows})
        },
        "matched": matched,
        "injected_matched_macro_auroc": float(
            np.mean([v["auroc"] for k, v in matched.items() if k.startswith("inj_")])
        ),
        "threshold_rule": "strict > cutoff; cutoff fitted on all fitting negatives",
    }


def bootstrap(rows: list[dict], scores: dict[str, np.ndarray]) -> dict:
    """Compare the prespecified two-score arm to safety-only calibration."""
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["task_id"]].append(i)
    tasks = sorted(groups)
    labels = np.array([r["ground_truth"] for r in rows])
    # Include only tasks with both behavior labels for each injected template.
    matched = {}
    for template in TEMPLATES:
        valid = {
            t
            for t in tasks
            if {
                rows[i]["ground_truth"]
                for i in groups[t]
                if rows[i]["condition"] == f"inj_{template}"
            }
            == {0, 1}
        }
        matched[template] = np.array(
            [
                r["task_id"] in valid and r["condition"] == f"inj_{template}"
                for r in rows
            ]
        )
    rng = np.random.default_rng(0)
    deltas = []
    for _ in range(500):
        indices = np.concatenate(
            [groups[tasks[i]] for i in rng.integers(len(tasks), size=len(tasks))]
        )
        values = []
        for arm in ("safety_calibrated", "two_score"):
            aucs = []
            for template in TEMPLATES:
                subset = indices[matched[template][indices]]
                aucs.append(roc_auc_score(labels[subset], scores[arm][subset]))
            values.append(np.mean(aucs))
        deltas.append(values[1] - values[0])
    return {
        "repeats": 500,
        "seed": 0,
        "task_groups": len(tasks),
        "two_score_minus_safety_calibrated_matched_auroc_ci95": np.quantile(
            deltas, [0.025, 0.975]
        ).tolist(),
    }


def main() -> None:
    config = json.loads((HERE / "config.json").read_text())
    if config["arms"] != list(ARMS) or (
        config["C"],
        config["max_iter"],
        config["random_state"],
    ) != (1, 1000, 0):
        raise ValueError("frozen fitting recipe drift")
    out = ROOT / "results/apps_score_fusion/fusion01"
    out.mkdir(parents=True, exist_ok=False)
    try:
        for name, sha in config["files_sha256"].items():
            if file_hash(ROOT / name) != sha:
                raise ValueError(f"input checksum drift: {name}")
        safety_root = ROOT / "results/b200-augmented-sdpa-apps01/evaluation"
        safety_receipt = json.loads((safety_root / "complete.json").read_text())
        if (
            not safety_receipt["parity_passed"]
            or safety_receipt["master_sha256"] != config["master_sha256"]
        ):
            raise ValueError("safety checkpoint/parity drift")
        safety = []
        for split in ("benchmark", "honest_controls"):
            path = safety_root / f"{split}.jsonl"
            if file_hash(path) != safety_receipt["files_sha256"][split]:
                raise ValueError("safety completion checksum drift")
            safety.extend(read_rows(path))
        presence_root = ROOT / "results/apps_injection_detection/presence01/sdpa"
        presence_receipt = json.loads((presence_root / "complete.json").read_text())
        merged = json.loads(
            (
                ROOT / "results/b200-augmented-sdpa-replay02/merged_artifact.json"
            ).read_text()
        )
        binding = json.loads((presence_root / "binding.json").read_text())
        if (
            not presence_receipt["parity_passed"]
            or binding["files_sha256"] != merged["files_sha256"]
        ):
            raise ValueError("presence checkpoint/parity drift")
        if (
            file_hash(presence_root / "predictions.jsonl")
            != presence_receipt["predictions_sha256"]
        ):
            raise ValueError("presence completion checksum drift")
        rows = join_scores(safety, read_rows(presence_root / "predictions.jsonl"))
        if len(rows) != 9114:
            raise ValueError("fusion population drift")
        partition = json.loads((ROOT / config["partition"]).read_text())
        fit_tasks, test_tasks = map(
            set, (partition["fit_tasks"], partition["test_tasks"])
        )
        if fit_tasks & test_tasks or fit_tasks | test_tasks != {
            r["task_id"] for r in rows
        }:
            raise ValueError("fusion task partition drift")
        fit = [r for r in rows if r["task_id"] in fit_tasks]
        test = [r for r in rows if r["task_id"] in test_tasks]
        if (len(fit), len(test)) != (896, 8218):
            raise ValueError("fit/test coverage drift")
        write_rows(out / "paired_inputs.jsonl", rows)
        write_json(
            out / "contract.json",
            {
                "config": config,
                "partition": partition,
                "fit_rows": len(fit),
                "test_rows": len(test),
                "sources": {
                    str(p.relative_to(ROOT)): file_hash(p)
                    for p in [
                        HERE / "run.py",
                        HERE / "config.json",
                        HERE / "README.md",
                        ROOT / "src/gleipnir/evaluation/score_fusion.py",
                    ]
                },
            },
        )
        models, reports, scores = {}, {}, {}
        for arm in ARMS:
            if arm == "safety_raw":
                fitted = np.array([r["safety_score"] for r in fit])
                scored = np.array([r["safety_score"] for r in test])
            else:
                models[arm] = fit_combiner(fit, arm)
                fitted, scored = (
                    combined_scores(pop, models[arm]) for pop in (fit, test)
                )
            threshold = fit_threshold(fit, fitted)
            predictions = [
                r | {"score": float(s)} for r, s in zip(test, scored, strict=True)
            ]
            write_rows(out / f"{arm}_predictions.jsonl", predictions)
            reports[arm] = report(predictions, threshold)
            scores[arm] = scored
        write_json(out / "models.json", models)
        write_json(
            out / "summary.json",
            {"arms": reports, "bootstrap": bootstrap(test, scores)},
        )
        write_json(
            out / "complete.json",
            {
                "passed": True,
                "rows": len(test),
                "files_sha256": {p.name: file_hash(p) for p in out.glob("*.json*")},
            },
        )
        print("fusion_complete", len(fit), len(test))
    except BaseException as error:
        write_json(
            out / "failure.json", {"type": type(error).__name__, "message": str(error)}
        )
        raise


if __name__ == "__main__":
    main()
