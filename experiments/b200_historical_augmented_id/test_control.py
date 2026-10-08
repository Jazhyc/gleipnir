"""Protect historical checkpoint identity and the evaluation-only wrapper."""

import json
from types import SimpleNamespace

import pytest

from experiments.b200_historical_augmented_id.run import (
    bind_completed_adapter,
    canonicalize_historical_predictions,
)
from gleipnir.data.monitoring import file_hash


@pytest.fixture
def completed(tmp_path):
    historical = tmp_path / "historical"
    for layout in ("causal_adapter", "model"):
        directory = historical / layout
        directory.mkdir(parents=True)
        (directory / "adapter_model.safetensors").write_bytes(layout.encode())
    metadata = historical / "causal_adapter/training_metadata.json"
    metadata.write_text('{"training": "historical BF16/SDPA"}')
    complete = historical / "complete.json"
    complete.write_text(
        json.dumps(
            {
                "status": "trained",
                "steps": 272,
                "master_sha256": file_hash(
                    historical / "causal_adapter/adapter_model.safetensors"
                ),
                "serving_sha256": file_hash(
                    historical / "model/adapter_model.safetensors"
                ),
            }
        )
    )
    paths = {
        "completed_training": complete,
        "completed_metadata": metadata,
        "completed_master": historical / "causal_adapter/adapter_model.safetensors",
        "completed_serving": historical / "model/adapter_model.safetensors",
    }
    return SimpleNamespace(
        root=tmp_path,
        config={"historical_adapter": "historical"},
        adapter=tmp_path / "run/4b/monitor",
        output=tmp_path / "run",
        input=lambda name: paths[name],
    )


def test_wrapper_preserves_original_weights_and_completion(completed):
    originals = {
        p: p.read_bytes()
        for p in (completed.root / "historical").rglob("*")
        if p.is_file()
    }
    bind_completed_adapter(completed)
    assert (completed.adapter / "complete.json").read_bytes() == completed.input(
        "completed_training"
    ).read_bytes()
    assert (completed.adapter / "causal_adapter").is_symlink()
    assert (completed.adapter / "model").is_symlink()
    assert all(p.read_bytes() == content for p, content in originals.items())
    assert (
        json.loads((completed.output / "historical_adapter.json").read_text())[
            "new_training"
        ]
        is False
    )


def test_changed_historical_master_fails_before_binding(completed):
    completed.input("completed_master").write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity"):
        bind_completed_adapter(completed)
    assert not completed.adapter.exists()


def test_different_existing_wrapper_is_never_rewired(completed):
    other = completed.root / "other"
    other.mkdir()
    completed.adapter.mkdir(parents=True)
    link = completed.adapter / "causal_adapter"
    link.symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError, match="different weights"):
        bind_completed_adapter(completed)
    assert link.resolve() == other.resolve()


def test_legacy_metadata_rewrap_keeps_scores_and_original_provenance():
    original = [
        {
            "id": "one",
            "ground_truth": 1,
            "source_dataset": "source",
            "prompt_sha256": "prompt",
            "prompt_tokens": 10,
            "score": 0.8,
            "original_metadata": {"older": "nesting"},
        }
    ]
    canonical = [
        {
            "id": "one",
            "metadata": {
                "ground_truth": 1,
                "source_dataset": "source",
                "original_metadata": {"newer": "nesting"},
            },
        }
    ]
    workload = [{"id": "one", "prompt_sha256": "prompt", "prompt_tokens": 10}]
    result = canonicalize_historical_predictions(original, canonical, workload)[0]
    assert result["score"] == 0.8
    assert result["historical_original_metadata"] == {"older": "nesting"}
    assert result["original_metadata"] == {"newer": "nesting"}
    assert original[0]["original_metadata"] == {"older": "nesting"}
    for key, value in (
        ("prompt_sha256", "different"),
        ("ground_truth", 0),
        ("source_dataset", "different"),
        ("id", "different"),
        ("prompt_tokens", 9),
    ):
        with pytest.raises(ValueError, match="identity"):
            canonicalize_historical_predictions(
                [{**original[0], key: value}], canonical, workload
            )
