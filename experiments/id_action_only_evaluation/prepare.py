"""Freeze cleaned-input provenance and prepare the shared evaluation runner."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.tool_trajectory_monitoring.benchmark_distilled_ood import (
    validate_config,
    validate_inputs,
    validate_jobs,
)
from experiments.tool_trajectory_monitoring.prepare_distillation_ood import (
    atomic_write_json,
    sha256_file,
)

ROOT = Path("results/id_action_only_evaluation")
CONFIG = Path("experiments/id_action_only_evaluation/config.json")
JOB = "soft-n21837-mixed-qwen35-4b-seed0"


def main() -> None:
    config = json.loads(CONFIG.read_text())
    scope = config["scope"]
    if sha256_file(Path(scope["input"])) != scope["input_sha256"]:
        raise ValueError("Cleaned ID input drift")
    manifest = {
        "output_sha256": scope["input_sha256"],
        "source_sha256": scope["source_input_sha256"],
        "student_template_sha256": config["prompt"]["template_sha256"],
        "transformation": "id-assistant-prose-and-thinking-removal-v1",
        "rows": 3012,
    }
    atomic_write_json(Path(scope["manifest"]), manifest)
    validate_config(config)
    validate_inputs(config)
    jobs = Path("results/tool_trajectory_distillation_mixed_qwen4b/lambda_jobs.jsonl")
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "jobs.jsonl").write_bytes(jobs.read_bytes())
    validate_jobs(config, "4b")
    if (ROOT / "status.json").exists():
        raise FileExistsError("Existing run status must not be reset")
    atomic_write_json(ROOT / "status.json", {"state": "prepared", "phase": "prepared"})
    print(f"Prepared {scope['rows']} cleaned ID rows: {sha256_file(CONFIG)}")


if __name__ == "__main__":
    main()
