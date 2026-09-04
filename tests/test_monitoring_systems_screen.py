import json
from pathlib import Path

import pytest

from gleipnir.monitoring_systems_screen import (
    load_config,
    prepare_screen,
    resolve_paths,
    round_robin_lanes,
    sha256_file,
    stable_stratified_selection,
    validate_prepared_artifacts,
)


def test_stable_selection_is_proportional_and_deterministic() -> None:
    records = [
        {
            "dataset": dataset,
            "index": index,
            "label": label,
        }
        for dataset in ("a", "b")
        for label in (0, 1)
        for index in range(4)
    ]
    first = stable_stratified_selection(records, count=8, seed=7)
    second = stable_stratified_selection(records, count=8, seed=7)
    assert first == second
    assert {
        (dataset, label): sum(
            row["dataset"] == dataset and row["label"] == label for row in first
        )
        for dataset in ("a", "b")
        for label in (0, 1)
    } == {("a", 0): 2, ("a", 1): 2, ("b", 0): 2, ("b", 1): 2}


def test_round_robin_lanes_never_share_a_gpu_concurrently() -> None:
    jobs = [{"job_name": f"job-{index}"} for index in range(5)]
    lanes = round_robin_lanes(jobs, gpu_count=2)
    assert [[job["job_name"] for job in lane] for lane in lanes] == [
        ["job-0", "job-2", "job-4"],
        ["job-1", "job-3"],
    ]


def test_prepared_manifest_detects_selection_drift(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    students = data_dir / "students.jsonl"
    soft_targets = data_dir / "soft.jsonl"
    rows = [
        {
            "dataset": "source",
            "source_dataset": "source",
            "index": index,
            "label": index % 2,
            "student_direct_tokens": index + 1,
            "trajectory_sha256": f"{index:064x}",
        }
        for index in range(4)
    ]
    students.write_text("".join(json.dumps(row) + "\n" for row in rows))
    soft_targets.write_text("{}\n" * 4)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "systems_screen_schema": 1,
                "campaign_id": "test-screen",
                "gpus": 2,
                "artifacts": {"result_dir": "unused"},
                "data": {
                    "directory": "unused",
                    "student_rows": students.name,
                    "soft_targets": soft_targets.name,
                    "rows": 4,
                    "student_rows_sha256": sha256_file(students),
                    "soft_targets_sha256": sha256_file(soft_targets),
                },
                "selection": {"rows": 2, "seed": 0, "preflight_rows": 1},
                "recipe": {},
                "conditions": [
                    {"job_name": "control", "design_role": "control"},
                    {"job_name": "candidate", "design_role": "candidate"},
                ],
                "preflight": {"condition": "candidate"},
                "comparison": {
                    "control": "control",
                    "candidate": "candidate",
                },
            }
        )
    )
    result_dir = tmp_path / "results"
    prepare_screen(config_path, data_dir=data_dir, result_dir=result_dir)
    config = load_config(config_path)
    paths = resolve_paths(config, data_dir=data_dir, result_dir=result_dir)
    validate_prepared_artifacts(config_path, config, paths)
    paths.selection.write_text(paths.selection.read_text() + "{}\n")
    with pytest.raises(ValueError, match="selection_sha256"):
        validate_prepared_artifacts(config_path, config, paths)
