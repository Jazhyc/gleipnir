"""Protect the fixed OOD selection and unchanged prompt/data contract."""

import json
from pathlib import Path

from experiments.monitoring_mil_ood.run import JOB, build_config
from experiments.tool_trajectory_monitoring.benchmark_distilled_ood import (
    validate_config,
)
from gleipnir.monitoring_systems_screen import atomic_write_jsonl


def test_frozen_selection_preserves_ood_contract(tmp_path):
    template = json.loads(
        Path(
            "experiments/tool_trajectory_monitoring/distillation_ood_benchmark.json"
        ).read_text()
    )
    original = json.loads(json.dumps(template))
    jobs = tmp_path / "jobs.jsonl"
    atomic_write_jsonl(jobs, [{"job_name": JOB}])
    config = build_config(template, jobs)
    validate_config(config)
    assert template == original
    for key in ("scope", "prompt", "scoring"):
        assert config[key] == original[key]
    assert list(config["model_groups"]) == ["4b"]
    group = config["model_groups"]["4b"]
    assert group["expected_jobs"] == [JOB]
    assert group["parity_job"] == JOB
    assert group["evaluate_base"] is False
    assert config["engine"]["tensor_parallel_size"] == 1
