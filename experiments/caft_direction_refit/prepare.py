"""Bind the completed CAFT checkpoint to the existing APPS refit protocol."""

import json
from pathlib import Path

from gleipnir.data.monitoring import file_hash, write_json

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def binding(path: str) -> dict:
    return {"path": path, "sha256": file_hash(ROOT / path)}


def prepare() -> None:
    startup = json.loads(
        (ROOT / "experiments/activation_filter_projection/startup.json").read_text()
    )
    previous = "results/caft-regular-sdpa02"
    paths = {
        "adapter_complete": f"{previous}/4b/monitor/complete.json",
        "canary": f"{previous}/canary_workload.jsonl",
        "compiled_predictions": f"{previous}/optimized_canary_predictions.json",
        "master_reference": f"{previous}/4b/monitor/parity_reference.json",
        "merge_parity": f"{previous}/merged_parity.json",
        "merged_artifact": f"{previous}/merged_artifact.json",
        "parent_server": f"{previous}/server.json",
        "lens_staging": "results/b200_vllm_lens/staging.json",
    }
    startup["inputs"] = {k: binding(p) for k, p in paths.items()}
    startup["model"] = "/tmp/gleipnir-merged/caft-regular-sdpa02"
    write_json(HERE / "startup.json", startup)
    original = "results/b200_injection_direction/sdpa_injection_direction01"
    filtered = "results/activation_filter_refit/ranked20_refit01"
    paths = {
        "fitting": f"{original}/fit_workload.jsonl",
        "testing": f"{original}/apps_workload.jsonl",
        "partition": f"{original}/partition.json",
        "original_directions": f"{original}/directions.npz",
        "filtered_directions": f"{filtered}/directions.npz",
        "original_summary": f"{original}/summary.json",
        "filtered_summary": f"{filtered}/summary.json",
        "compiled_benchmark": f"{previous}/evaluation/benchmark.jsonl",
        "compiled_controls": f"{previous}/evaluation/honest_controls.jsonl",
        "startup": "experiments/caft_direction_refit/startup.json",
    }
    write_json(
        HERE / "config.json",
        {
            "campaign_id": "caft_refit01",
            "startup_name": "caft_refit01",
            "direction_layer": 20,
            "random_seed": 0,
            "clamp_beta": 0.25,
            "batch_rows": 128,
            "concurrency": 64,
            "capture_batch_rows": 32,
            "capture_concurrency": 8,
            "promote": False,
            "inputs": {k: binding(p) for k, p in paths.items()},
        },
    )


if __name__ == "__main__":
    prepare()
