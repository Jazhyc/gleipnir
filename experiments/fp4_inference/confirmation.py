"""Freeze trajectory exclusions and report broader-split confirmation metrics."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from experiments.fp4_inference.repeat_analysis import ROW_FIELDS
from experiments.fp4_inference.run import resolve_runtime_config
from experiments.local_inference.compare_runs import compare_runs
from experiments.local_inference.core import compare, read_rows, write_json
from gleipnir.binary_evaluation import metric_views
from gleipnir.qwen35_adapter_rebase import sha256_file


def heldout_ids(full: list[dict], used: list[dict]) -> list[str]:
    """Exclude whole connected groups of original/transformed trajectory hashes."""
    if len({r["id"] for r in full}) != len(full):
        raise ValueError("Duplicate confirmation IDs")
    excluded = {r["id"] for r in used}
    hashes = {
        r[k] for r in used for k in ("trajectory_sha256", "original_trajectory_sha256")
    }
    changed = True
    while changed:
        changed = False
        for row in full:
            lineage = {row["trajectory_sha256"], row["original_trajectory_sha256"]}
            if row["id"] in excluded or lineage & hashes:
                if row["id"] not in excluded or not lineage <= hashes:
                    changed = True
                excluded.add(row["id"])
                hashes.update(lineage)
    return [r["id"] for r in full if r["id"] not in excluded]


def freeze(output: Path, configs: dict[str, Path] | None = None) -> dict:
    """Freeze exclusions independently of future candidate scores or metrics."""
    sources = {
        "input": Path("data/local_inference/subset.jsonl"),
        "development": Path("data/local_inference/iteration32.jsonl"),
        "capture": Path("results/fp4_inference/capture_all/manifest.json"),
        "reference": Path("results/local_inference/reference.json"),
    }
    if configs is not None:
        if set(configs) != {"baseline_config", "candidate_config"}:
            raise ValueError("Selection requires both serving configurations")
        sources.update(configs)
    full, development = [read_rows(sources[k]) for k in ("input", "development")]
    capture = json.loads(sources["capture"].read_text())
    reference = json.loads(sources["reference"].read_text())
    if capture["subset_sha256"] != sha256_file(sources["input"]) or reference[
        "subset_sha256"
    ] != sha256_file(sources["input"]):
        raise ValueError("Confirmation parent identity mismatch")
    if capture["iteration32_sha256"] != sha256_file(sources["development"]):
        raise ValueError("Confirmation development identity mismatch")
    canaries = [r for r in full if r["id"] in reference["ids"]]
    if len(canaries) != len(reference["ids"]):
        raise ValueError("Missing confirmation canary identities")
    selected = heldout_ids(full, development + capture["rows"] + canaries)
    selected_set = set(selected)
    manifest = {
        "source_files": {
            k: {"path": str(p), "sha256": sha256_file(p)} for k, p in sources.items()
        },
        "analysis_sha256": sha256_file(Path(__file__)),
        "full_rows": len(full),
        "used_counts": {
            "development": len(development),
            "capture": len(capture["rows"]),
            "canaries": len(canaries),
        },
        "heldout_ids": selected,
        "heldout_rows": len(selected),
        "excluded_ids": [r["id"] for r in full if r["id"] not in selected_set],
        "heldout_source_label_counts": dict(
            Counter(
                f"{r['source']}:{r['label']}" for r in full if r["id"] in selected_set
            )
        ),
        "rule": (
            "Exclude used IDs and connected original/transformed trajectory hashes. "
            "No selection using confirmation scores."
        ),
    }
    if output.exists():
        raise FileExistsError(output)
    write_json(output, manifest)
    return manifest


def report(baseline: Path, candidate: Path, manifest_path: Path) -> dict:
    """Audit the full paired pass and compute only the frozen held-out metrics."""
    manifest = json.loads(manifest_path.read_text())
    if manifest["analysis_sha256"] != sha256_file(Path(__file__)):
        raise ValueError("Frozen confirmation analysis changed")
    for source in manifest["source_files"].values():
        if sha256_file(Path(source["path"])) != source["sha256"]:
            raise ValueError("Frozen confirmation input changed")
    for name, root in (("baseline_config", baseline), ("candidate_config", candidate)):
        if name in manifest["source_files"]:
            source = Path(manifest["source_files"][name]["path"])
            expected = resolve_runtime_config(json.loads(source.read_text()))
            launched = json.loads((root / "launch_config.json").read_text())
            if launched != expected:
                raise ValueError("Serving configuration differs from frozen selection")
    full = read_rows(Path(manifest["source_files"]["input"]["path"]))
    development = read_rows(Path(manifest["source_files"]["development"]["path"]))
    capture = json.loads(Path(manifest["source_files"]["capture"]["path"]).read_text())
    reference = json.loads(
        Path(manifest["source_files"]["reference"]["path"]).read_text()
    )
    canaries = [r for r in full if r["id"] in reference["ids"]]
    expected_ids = heldout_ids(full, development + capture["rows"] + canaries)
    if (
        expected_ids != manifest["heldout_ids"]
        or len(expected_ids) != manifest["heldout_rows"]
    ):
        raise ValueError("Frozen trajectory exclusion mismatch")
    paired = compare_runs(baseline, candidate)
    if paired["runner_audit"]["changed"]:
        raise ValueError("Confirmation score runner changed")
    selected = set(manifest["heldout_ids"])
    arrays = []
    for root in (baseline, candidate):
        predictions = json.loads((root / "predictions_0.json").read_text())
        for row, prediction in zip(full, predictions, strict=True):
            if any(row[k] != prediction[k] for k in ROW_FIELDS):
                raise ValueError("Prediction differs from frozen confirmation input")
        arrays.append([r for r in predictions if r["id"] in selected])
    metrics = [
        metric_views(pd.DataFrame(rows).rename(columns={"source": "dataset"}))
        for rows in arrays
    ]
    qa, qb = [m["macro"]["macro"] for m in metrics]
    auroc_loss, brier_increase = qa["auroc"] - qb["auroc"], qb["brier"] - qa["brier"]
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "full_pair": {k: v for k, v in paired.items() if k != "paired"},
        "heldout_rows": len(arrays[0]),
        "heldout_metrics": {"baseline": metrics[0], "candidate": metrics[1]},
        "heldout_drift": compare(
            [r["score"] for r in arrays[0]], [r["score"] for r in arrays[1]]
        ),
        "heldout_drift_by_source": {
            source: compare(
                [r["score"] for r in arrays[0] if r["source"] == source],
                [r["score"] for r in arrays[1] if r["source"] == source],
            )
            for source in sorted({r["source"] for r in arrays[0]})
        },
        "heldout_quality_gate": {
            "macro_auroc_loss": auroc_loss,
            "macro_brier_increase": brier_increase,
            "passed": auroc_loss <= 0.01 and brier_increase <= 0.01,
        },
        "confirmation_gates_passed": paired["candidate_serving_parity_passed"]
        and auroc_loss <= 0.01
        and brier_increase <= 0.01,
        "note": (
            "One frozen candidate; no tuning or fallback selection on confirmation. "
            "Within-split evidence only."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--baseline-config", type=Path)
    parser.add_argument("--candidate-config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.freeze:
        if bool(args.baseline_config) != bool(args.candidate_config):
            parser.error("Selection requires both baseline and candidate configs")
        configs = (
            {
                "baseline_config": args.baseline_config,
                "candidate_config": args.candidate_config,
            }
            if args.baseline_config
            else None
        )
        result = freeze(args.output, configs)
    else:
        if not all((args.baseline, args.candidate, args.manifest)):
            parser.error("Reporting requires baseline, candidate and manifest")
        if args.output.exists():
            raise FileExistsError(args.output)
        result = report(args.baseline, args.candidate, args.manifest)
        write_json(args.output, result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in result
                if k in {"heldout_rows", "heldout_quality_gate"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
