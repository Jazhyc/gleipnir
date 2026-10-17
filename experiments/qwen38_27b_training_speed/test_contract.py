import json
from pathlib import Path

from experiments.qwen38_27b_training_speed.prepare import MODEL, SOURCE, job_27b


def test_27b_job_changes_only_model_paths_and_checkpointing() -> None:
    path = SOURCE / "jobs.jsonl"
    if not path.exists():
        import pytest

        pytest.skip("frozen 4B jobs are not materialized")
    original = json.loads(Path(path).read_text().splitlines()[0])
    job = job_27b(original, "runs")
    changed = {k for k in set(job) | set(original) if job.get(k) != original.get(k)}
    assert changed <= {
        "model",
        "model_revision",
        "gradient_checkpointing",
        "gradient_checkpointing_policy",
        "gradient_checkpointing_layer_indices",
        "nonreentrant_checkpointing",
        "output_dir",
        "causal_adapter_dir",
        "model_dir",
        "hydra_log_dir",
    }
    assert job["model"] == MODEL
    assert job["gradient_checkpointing_policy"] == "all"
    for key in ("selection_sha256", "soft_targets_sha256", "student_rows_sha256"):
        assert job[key] == original[key]
