"""Bind unchanged regular data and the previously frozen injection axis."""

from pathlib import Path

import yaml

from gleipnir.data.monitoring import file_hash

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def prepare() -> None:
    config = yaml.safe_load(
        (ROOT / "experiments/activation_filter_training/ranked.yaml").read_text()
    )
    names = {
        "id",
        "id_manifest",
        "id_workload",
        "benchmark",
        "honest_controls",
        "benchmark_workload",
        "honest_controls_workload",
        "startup_reference",
        "serving_selection",
        "initial_adapter_config",
        "initial_weights",
    }
    config["inputs"] = {k: v for k, v in config["inputs"].items() if k in names}
    paths = {
        "training": "data/student_injection_awareness/regular/student_rows.jsonl",
        "teacher": "data/student_injection_awareness/soft_targets.jsonl",
        ("token_audit"): (
            "results/student_injection_awareness/4b/regular/token_audit.json"
        ),
        ("concept_direction"): (
            "results/b200_injection_direction/sdpa_injection_direction01/directions.npz"
        ),
        ("direction_partition"): (
            "results/b200_injection_direction/sdpa_injection_direction01/partition.json"
        ),
        ("baseline_id"): (
            "results/b200_attention_precision/precision01/fp8/id_predictions.json"
        ),
        "baseline_apps": "results/b200_apps/apps02/summary.json",
        "baseline_benchmark": "results/b200_apps/apps02/benchmark.jsonl",
        "baseline_controls": "results/b200_apps/apps02/honest_controls.jsonl",
    }
    for name, path in paths.items():
        config["inputs"][name] = {"path": path, "sha256": file_hash(ROOT / path)}
    config.update(
        campaign_id="caft-regular-sdpa02",
        training_rows=8688,
        injected_rows=0,
        expected_steps=272,
        expected_training_tokens=83816369,
        concept_ablation=True,
        augmentation_audit=False,
        merged_model="/tmp/gleipnir-merged/caft-regular-sdpa02",
        hypothesis=(
            "Regular-data CAFT reduces APPS injected-honest alarms withou"
            "t synthetic augmentation while retaining ID ranking and back"
            "door recall."
        ),
        intervention=(
            "Project the original unfiltered-model unit layer-20 injectio"
            "n axis out after every decoder layer/all tokens during train"
            "ing and backward; remove hooks for inference."
        ),
        selection=(
            "Final complete one-epoch checkpoint only. Fixed axis/geometr"
            "y/seed; no ID/APPS tuning or promotion."
        ),
        stop_condition=(
            "Stop on provenance/runtime drift, nonfinite/missing gradient"
            "s, failed operator or layer/token coverage, OOM, or new-adap"
            "ter parity/native failure."
        ),
    )
    config["baselines"] = {
        "cached_regular_id": {
            "kind": "id",
            "input": "baseline_id",
            ("qualification"): (
                "User-requested cached regular FP4-trained optimized-serving "
                "control; training precision, serving and host differ from th"
                "is BF16/SDPA CAFT experiment."
            ),
        },
        "cached_regular_apps": {
            "kind": "apps",
            "input": "baseline_apps",
            "keys": [],
            ("qualification"): (
                "User-requested cached regular FP4-trained optimized APPS con"
                "trol; backend/host/precision differences retained, no new co"
                "ntrol training."
            ),
        },
    }
    config["extra_sources"] = [
        str(p.relative_to(ROOT)) for p in HERE.glob("*.*") if p.name != "config.yaml"
    ]
    (HERE / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))


if __name__ == "__main__":
    prepare()
