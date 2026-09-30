import json
from pathlib import Path

import pytest

from gleipnir.monitoring_systems_screen import (
    add_missing_token_lengths,
    compiler_cache_directory,
    load_config,
    prepare_screen,
    require_frozen_config,
    resolve_paths,
    round_robin_lanes,
    selection_manifest_row,
    sha256_file,
    stable_stratified_selection,
    validate_prepared_artifacts,
)


def test_shared_compiler_cache_is_explicit_and_separate_for_each_gpu(
    tmp_path: Path,
) -> None:
    config = load_config(Path("experiments/b200_training_throughput/config.yaml"))
    paths = resolve_paths(config, result_dir=tmp_path / "results")
    assert compiler_cache_directory(config, paths, "control", 0) == (
        paths.result_dir / "compile_cache" / "control"
    )
    config["compiler_cache_root"] = str(tmp_path / "shared-cache")
    assert compiler_cache_directory(config, paths, "control", 0) == (
        compiler_cache_directory(config, paths, "candidate", 0)
    )
    assert compiler_cache_directory(config, paths, "control", 0) != (
        compiler_cache_directory(config, paths, "control", 1)
    )
    config["compiler_cache_root"] = ""
    with pytest.raises(ValueError, match="nonempty"):
        compiler_cache_directory(config, paths, "control", 0)


def test_mixed_source_length_inference_preserves_prompts_and_available_provenance():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            return messages[0]["content"]

        def encode(self, prompt, **kwargs):
            assert kwargs["add_special_tokens"] is False
            return list(prompt)

    rows = [
        {
            "dataset": "monitoring",
            "index": "a",
            "label": 1,
            "student_prompt": "original",
            "student_direct_tokens": 99,
            "trajectory_sha256": "preserved",
        },
        {"dataset": "deception", "index": 2, "label": 0, "student_prompt": "cached"},
    ]
    assert add_missing_token_lengths(rows, Tokenizer()) == 1
    assert rows[0]["student_direct_tokens"] == 99
    assert rows[1]["student_prompt"] == "cached"
    assert rows[1]["student_direct_tokens"] == len("cachedPrediction:")
    selected = selection_manifest_row(rows[1])
    assert selected["index"] == 2 and selected["label"] == 0
    assert "trajectory_sha256" not in selected
    assert selection_manifest_row(rows[0])["trajectory_sha256"] == "preserved"


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
    manifest = prepare_screen(config_path, data_dir=data_dir, result_dir=result_dir)
    frozen_config = Path(manifest["config"])
    config = load_config(frozen_config)
    paths = resolve_paths(config)
    assert paths.data_dir == data_dir.resolve()
    assert paths.result_dir == result_dir.resolve()
    assert manifest["path_overrides"] == {
        "data_dir": data_dir.resolve().as_posix(),
        "result_dir": result_dir.resolve().as_posix(),
    }
    validate_prepared_artifacts(frozen_config, config, paths)
    paths.selection.write_text(paths.selection.read_text() + "{}\n")
    with pytest.raises(ValueError, match="selection_sha256"):
        validate_prepared_artifacts(frozen_config, config, paths)


def test_run_contract_requires_named_frozen_snapshot(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="resolved_config.json"):
        require_frozen_config(tmp_path / "config.yaml")
    require_frozen_config(tmp_path / "resolved_config.json")
