"""Census filtering, exact teacher joins and matched evaluation scope."""

from __future__ import annotations

import copy
import json
import math
from types import SimpleNamespace

import pytest

from experiments.monitoring_injection_removal.evaluate import evaluation_cells
from experiments.monitoring_injection_removal.train import make_job
from gleipnir.monitoring_exclusions import census_filter


def population():
    rows, targets, predictions = [], [], []
    for index, margin in enumerate((-1.0, 0.0, 1.0)):
        row = {
            "dataset": "tool_trajectory/source",
            "index": str(index),
            "label": index % 2,
            "source_dataset": "source",
            "lineage_group": f"group:{index}",
            "trajectory_sha256": f"trace-{index}",
            "student_prompt_sha256": f"student-{index}",
            "teacher_rendered_prompt_sha256": f"teacher-{index}",
        }
        rows.append(row)
        targets.append(
            {
                "dataset": row["dataset"],
                "index": str(index),
                "label": row["label"],
                "soft_target": 0.2,
                "rendered_prompt_sha256": f"teacher-{index}",
                "provider": "frozen-provider",
            }
        )
        predictions.append(
            {
                "id": str(index),
                "score": 1 / (1 + math.exp(-margin)),
                "logprob_0": -2.0,
                "logprob_1": -2.0 + margin,
                "source": "source",
                "label": row["label"],
                "lineage_group": row["lineage_group"],
                "trajectory_sha256": row["trajectory_sha256"],
                "student_prompt_sha256": row["student_prompt_sha256"],
                "contract_sha256": "frozen",
            }
        )
    return rows, targets, predictions


def test_threshold_includes_ties_and_preserves_teacher_records():
    rows, targets, predictions = population()
    snapshot = copy.deepcopy((rows, targets, predictions))
    keep, teacher, removed = census_filter(
        rows, targets, predictions[::-1], threshold=0.5, contract_sha256="frozen"
    )
    assert keep == rows[:1] and keep[0] is rows[0]
    assert teacher == targets[:1] and teacher[0] is targets[0]
    assert {r["id"] for r in removed} == {"1", "2"}
    assert (rows, targets, predictions) == snapshot


@pytest.mark.parametrize(
    "drift",
    [
        "duplicate",
        "missing",
        "foreign",
        "trajectory",
        "label",
        "contract",
        "nonfinite",
        "score",
        "teacher",
    ],
)
def test_filter_fails_closed_on_population_or_provenance_drift(drift):
    rows, targets, predictions = population()
    if drift == "duplicate":
        predictions[-1] = predictions[0]
    elif drift == "missing":
        predictions.pop()
    elif drift == "foreign":
        predictions[0]["id"] = "foreign"
    elif drift == "trajectory":
        predictions[0]["trajectory_sha256"] = "changed"
    elif drift == "label":
        predictions[0]["label"] = 1
    elif drift == "contract":
        predictions[0]["contract_sha256"] = "changed"
    elif drift == "nonfinite":
        predictions[0]["score"] = math.nan
    elif drift == "score":
        predictions[0]["score"] = 0.7
    else:
        targets.pop()
    with pytest.raises(ValueError):
        census_filter(
            rows, targets, predictions, threshold=0.5, contract_sha256="frozen"
        )


def test_id_uses_only_standard_prompt_and_grid_uses_three():
    cells = evaluation_cells()
    assert [(s, p) for s, p in cells if s == "id"] == [("id", "neutral")]
    assert len(cells) == 7
    assert {p for s, p in cells if s == "benchmark"} == {
        "neutral",
        "aggressive",
        "conservative",
    }


def test_training_keeps_recipe_and_duration_after_filtering():
    config = {
        "model": {"id": "pinned", "revision": "revision"},
        "seed": 0,
        "learning_rate": 5e-5,
        "startup_validation_reference": "reference",
    }
    job = make_job(config, {"sequence_packing": True, "max_length": 29696})
    assert job["num_train_epochs"] == 1 and job["max_steps"] == -1
    assert job["sequence_packing"] and job["max_length"] == 29696
    assert job["soft_loss_weight"] == 1 and job["direct_loss_weight"] == 0
    assert not job["gradient_checkpointing"]
    assert "monitoring_injection_removal" in job["output_dir"]


def test_training_validation_accepts_new_update_count_and_rejects_old_count(
    monkeypatch,
):
    from experiments.monitoring_hard_labels import train

    monkeypatch.setattr(
        train, "resolved_profile", lambda: {"metadata_expectations": {}}
    )
    metadata = {
        "sequence_packing": {
            "initial_master_sha256": "initial",
            "final_master_sha256": "changed",
            **{
                k: {"passed": True}
                for k in ("eager_canary", "compiled_canary", "preflight")
            },
        },
        "training_state": {"global_step": 236},
        "losses": {
            "soft_weight": 1.0,
            "direct_weight": 0.0,
            "completion_weight": 0.0,
            "accumulation_policy": "sum_per_example_over_logical_batch_v1",
        },
        "gradient_checkpointing": False,
        "quantization": {"enabled": False},
        "direct_logits_mode": "selected_positions",
        "optimization": {"learning_rate": 5e-5},
        "gated_delta_backend": {"backend": "flashqla"},
    }
    job = {"soft_loss_weight": 1.0, "direct_loss_weight": 0.0, "learning_rate": 5e-5}
    train.validate_training_metadata(metadata, job, "initial", expected_steps=236)
    with pytest.raises(ValueError, match="completed training"):
        train.validate_training_metadata(metadata, job, "initial")


def test_completed_predictions_reject_changed_adapter_or_bytes(tmp_path, monkeypatch):
    from experiments.monitoring_injection_removal import evaluate
    from gleipnir.monitoring_campaign_data import (
        digest,
        file_hash,
        write_json,
        write_rows,
    )

    def canonical(value):
        return digest(json.dumps(value, sort_keys=True))

    monkeypatch.setattr(evaluate, "EVAL_DATA", tmp_path / "inputs")
    monkeypatch.setattr(evaluate, "DEST", tmp_path / "results")
    grid = SimpleNamespace(digest=canonical, validate_saved=lambda *x: None)
    monkeypatch.setattr(
        evaluate.baseline, "benchmark_tools", lambda: (None, grid, None, None)
    )
    inputs = tmp_path / "inputs/id/neutral.jsonl"
    path = tmp_path / "results/filtered/id_neutral.jsonl"
    write_rows(inputs, [{"id": "row", "metadata": {}, "prompt": "fixed"}])
    write_rows(path, [{"id": "row", "score": 0.3}])
    identity = {"adapter": "A"}
    spec = {
        "identity": identity,
        "split": "id",
        "prompt": "neutral",
        "input_sha256": file_hash(inputs),
    }
    contract = canonical(spec)
    write_json(path.with_suffix(".contract.json"), {"sha256": contract, **spec})
    write_json(
        path.with_suffix(".complete.json"),
        {
            "passed": True,
            "rows": 1,
            "sha256": file_hash(path),
            "contract_sha256": contract,
        },
    )
    evaluate.completed_cell("id", "neutral", identity)
    with pytest.raises(ValueError, match="identity drift"):
        evaluate.completed_cell("id", "neutral", {"adapter": "B"})
    write_rows(path, [{"id": "row", "score": 0.8}])
    with pytest.raises(ValueError, match="identity drift"):
        evaluate.completed_cell("id", "neutral", identity)
