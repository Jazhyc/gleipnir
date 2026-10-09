"""Compare frozen conditions on APPS tasks excluded from direction fitting."""

from __future__ import annotations

import json
from pathlib import Path

from gleipnir.data.monitoring import file_hash, read_rows, write_json
from gleipnir.evaluation.apps import summarize_apps

ROOT = Path(__file__).resolve().parents[2]
PARTITION = (
    ROOT / "results/b200_injection_direction/sdpa_injection_direction01/partition.json"
)


def heldout_apps(benchmark: list[dict], controls: list[dict], fit: set[str]) -> dict:
    """Exclude every fitting-task variant and retain exact held-out lineage."""
    b = [r for r in benchmark if str(r["task_id"]) not in fit]
    c = [r for r in controls if str(r["task_id"]) not in fit]
    tasks = {str(r["task_id"]) for r in b + c}
    if len(tasks) != 591 or len(b) + len(c) != 8218:
        raise ValueError("held-out APPS population drift")
    if any(str(r["task_id"]) in fit for r in b + c):
        raise ValueError("direction-fit task leakage")
    result = summarize_apps(b, c, expected_clean=591, expected_backdoors=583)
    conditions = [
        result["conditions"][f"inj_{name}"]
        for name in ["transfer_gpt4", "transfer_gpt41", "transfer_gpt4o"]
    ]
    result.update(
        mean_injected_backdoor_tpr=sum(r["backdoor_tpr"] for r in conditions) / 3,
        mean_matched_auroc=sum(r["matched"]["auroc"] for r in conditions) / 3,
        mean_matched_pauroc_at_20=sum(r["matched"]["pauroc_at_20"] for r in conditions)
        / 3,
    )
    return result


def analyze() -> None:
    partition = json.loads(PARTITION.read_text())
    fit = set(partition["fit_tasks"])
    reports = {}
    folders = {
        "unfiltered": "b200-augmented-sdpa-apps01",
        "ranked": "activation-filter-ranked20-sdpa01",
        "random": "activation-filter-matched20-sdpa01",
    }
    for name, folder in folders.items():
        root = ROOT / "results" / folder
        completion = json.loads((root / "evaluation/complete.json").read_text())
        if completion.get("rows") != (9114 if name == "unfiltered" else 12126):
            raise ValueError("incomplete evaluation")
        for split in ["benchmark", "honest_controls"]:
            if (
                file_hash(root / f"evaluation/{split}.jsonl")
                != completion["files_sha256"][split]
            ):
                raise ValueError("evaluation checksum drift")
        reports[name] = heldout_apps(
            read_rows(root / "evaluation/benchmark.jsonl"),
            read_rows(root / "evaluation/honest_controls.jsonl"),
            fit,
        )
    keys = [
        "mean_injected_honest_fpr",
        "mean_injected_backdoor_tpr",
        "mean_matched_auroc",
        "mean_matched_pauroc_at_20",
    ]
    # Preserve all metrics even if a future metric schema omits a compact field.
    comparisons = {
        f"ranked_minus_{control}": {
            k: reports["ranked"][k] - reports[control][k]
            for k in keys
            if k in reports["ranked"] and k in reports[control]
        }
        for control in ["unfiltered", "random"]
    }
    write_json(
        ROOT / "results/activation_filter_training/heldout_apps.json",
        {
            "direction_partition_sha256": file_hash(PARTITION),
            "fit_tasks_excluded": sorted(fit),
            "heldout_tasks": 591,
            "metrics": reports,
            "differences": comparisons,
            "qualification": (
                "Direction-fitting APPS tasks are excluded from primary results. "
                "Registered full-population summaries are diagnostics. One seed; "
                "no robust causal training attribution or promotion."
            ),
        },
    )
    print(json.dumps(comparisons), flush=True)


if __name__ == "__main__":
    analyze()
