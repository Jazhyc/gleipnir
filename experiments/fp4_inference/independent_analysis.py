"""Audit timing and numerical variation across frozen independent full starts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.fp4_inference.confirmation import report
from experiments.fp4_inference.repeat_analysis import (
    IDENTITY_FIELDS,
    ROW_FIELDS,
    load_run,
    stability,
)
from experiments.local_inference.core import write_json
from gleipnir.qwen35_adapter_rebase import sha256_file


def analyze(
    baselines: list[Path], candidates: list[Path], manifests: list[Path]
) -> dict:
    """Require unchanged configurations and identity for independent paired starts."""
    if not len(baselines) == len(candidates) == len(manifests) or len(baselines) < 2:
        raise ValueError("Require at least two matched independent start pairs")
    roots = baselines + candidates
    if len({p.resolve() for p in roots}) != len(roots):
        raise ValueError("Independent starts require distinct output paths")
    pairs = [
        report(a, b, m)
        for a, b, m in zip(baselines, candidates, manifests, strict=True)
    ]
    methods = {}
    for method, paths in (("baseline", baselines), ("candidate", candidates)):
        loaded = [load_run(p) for p in paths]
        first, rows, _, _ = loaded[0]
        selected = json.loads((paths[0] / "launch_config.json").read_text())
        selected.pop("output")
        for path, (result, current, scores, _) in zip(paths, loaded, strict=True):
            config = json.loads((path / "launch_config.json").read_text())
            config.pop("output")
            if config != selected:
                raise ValueError("Serving recipe changed between independent starts")
            if len(scores) != 1:
                raise ValueError("Expected one full pass in each independent engine")
            if any(result[k] != first[k] for k in IDENTITY_FIELDS):
                raise ValueError("Independent-start result identity mismatch")
            for a, b in zip(rows, current, strict=True):
                if any(a[k] != b[k] for k in ROW_FIELDS):
                    raise ValueError("Independent-start prediction identity mismatch")
        scores = np.stack([r[2][0] for r in loaded])
        margins = np.stack([r[3][0] for r in loaded])
        variation = stability(scores, margins)
        variation["independent_starts"] = variation.pop("passes")
        times = [r[0]["median_seconds"] for r in loaded]
        methods[method] = {
            "paths": [str(p) for p in paths],
            "result_sha256": [sha256_file(p / "result.json") for p in paths],
            "seconds": times,
            "median_seconds": float(np.median(times)),
            "minimum_seconds": min(times),
            "maximum_seconds": max(times),
            "variation": variation,
            "all_original_canaries_passed": all(
                r[0]["serving_parity_passed"] for r in loaded
            ),
        }
    return {
        "analysis_sha256": sha256_file(Path(__file__)),
        "manifest_sha256": [sha256_file(m) for m in manifests],
        "methods": methods,
        "median_speedup": methods["baseline"]["median_seconds"]
        / methods["candidate"]["median_seconds"],
        "paired_speedups": [p["full_pair"]["scoring_speedup"] for p in pairs],
        "heldout_quality_gates": [p["heldout_quality_gate"] for p in pairs],
        "all_confirmation_gates_passed": all(
            p["confirmation_gates_passed"] for p in pairs
        )
        and all(m["all_original_canaries_passed"] for m in methods.values()),
        "note": (
            "Identical frozen serving recipes; one full pass in each independent "
            "engine. Descriptive ranges only; no population equivalence claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", nargs="+", type=Path, required=True)
    parser.add_argument("--candidate", nargs="+", type=Path, required=True)
    parser.add_argument("--manifest", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = analyze(args.baseline, args.candidate, args.manifest)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
