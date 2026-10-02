"""Apply the predeclared ID rule to four completed, parity-gated conditions."""

from __future__ import annotations

import json
import math

from experiments.monitoring_hard_labels.prepare import (
    FRACTIONS,
    OUTPUT,
    configuration,
    verify_preparation,
    write_json,
)
from experiments.monitoring_hard_labels.train import validate_training_metadata


def select_winner(rows: list[dict], limits: dict) -> dict:
    if len(rows) != 4 or {r["variant"] for r in rows} != set(FRACTIONS):
        raise ValueError("summary does not cover the frozen four-cell grid")
    for row in rows:
        values = [row[k] for k in ("pauroc_at_20", "auroc", "brier")]
        values.extend(row["sources"].values())
        if set(row["sources"]) != {"gloom_exfiltration", "test_stride"}:
            raise ValueError("ID source coverage drift")
        if not all(math.isfinite(v) for v in values):
            raise ValueError("nonfinite summary metric")
    control = next(r for r in rows if r["variant"] == "hard000")
    eligible = []
    for row in rows:
        delta = {k: row[k] - control[k] for k in ("pauroc_at_20", "auroc", "brier")}
        source_delta = {s: v - control["sources"][s] for s, v in row["sources"].items()}
        row["control_deltas"] = {**delta, "sources": source_delta}
        row["eligible"] = (
            delta["pauroc_at_20"] >= limits["minimum_macro_pauroc_at_20_gain"]
            and min(source_delta.values())
            >= -limits["maximum_per_source_pauroc_at_20_regression"]
            and delta["brier"] <= limits["maximum_macro_brier_regression"]
        )
        if row["eligible"]:
            eligible.append(row)
    winner = max(
        eligible,
        key=lambda r: (r["pauroc_at_20"], r["auroc"], -r["brier"]),
        default=control,
    )
    return {
        "selected_variant": winner["variant"],
        "selected_hard_label_fraction": FRACTIONS[winner["variant"]],
        "control_retained": winner is control,
        "rule": limits,
        "one_seed_screen_only": True,
        "promotion": False,
        "ood_used": False,
    }


def main() -> None:
    config = configuration()
    verify_preparation()
    rows = []
    teachers = set()
    for variant, fraction in FRACTIONS.items():
        directory = OUTPUT / "4b" / variant
        complete = json.loads((directory / "complete.json").read_text())
        metadata = json.loads(
            (directory / "causal_adapter/training_metadata.json").read_text()
        )
        job = json.loads((directory / "job.json").read_text())
        validate_training_metadata(
            metadata, job, config["models"]["4b"]["initial_master_sha256"]
        )
        teachers.add(complete["teacher_targets_sha256"])
        parity = json.loads((directory / "serving_parity.json").read_text())
        result = json.loads((directory / "id/result.json").read_text())
        if not parity["passed"] or result["rows"] != 3012:
            raise ValueError("incomplete parity or ID evaluation")
        macro = result["metrics"]["macro"]
        rows.append(
            {
                "variant": variant,
                "hard_label_fraction": fraction,
                **{k: macro["macro"][k] for k in ("pauroc_at_20", "auroc", "brier")},
                "sources": {g["group"]: g["pauroc_at_20"] for g in macro["groups"]},
                "runtime": metadata["training_state"],
                "parity": parity["comparisons"],
            }
        )
    if len(teachers) != 1:
        raise ValueError("teacher targets differ across cells")
    selection = select_winner(rows, config["selection"])
    write_json(OUTPUT / "summary.json", {"candidates": rows, "selection": selection})
    print(json.dumps(selection, indent=2), flush=True)


if __name__ == "__main__":
    main()
