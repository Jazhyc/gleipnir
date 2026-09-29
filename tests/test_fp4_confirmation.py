"""Confirmation holds out complete connected trajectory groups."""

import json
from pathlib import Path

import pytest

from experiments.fp4_inference import confirmation
from experiments.fp4_inference.confirmation import heldout_ids
from gleipnir.qwen35_adapter_rebase import sha256_file


@pytest.mark.parametrize(
    "source,target",
    [
        ("baseline", "bf16_full_confirmation"),
        ("fp8_mlp_b8192_confirm", "fp8_mlp_b8192_full_confirmation"),
    ],
)
def test_full_confirmation_preserves_selected_serving_configuration(source, target):
    root = Path("experiments/fp4_inference/configs")
    selected = json.loads((root / f"{source}.json").read_text())
    full = json.loads((root / f"{target}.json").read_text())
    assert full.pop("input") == "data/local_inference/subset.jsonl"
    assert full.pop("manifest") == "data/local_inference/manifest.json"
    assert full.pop("repeats") == 1
    assert full.pop("subset_counts") == {
        "test_stride:0": 63,
        "test_stride:1": 98,
        "gloom_exfiltration:0": 176,
        "gloom_exfiltration:1": 175,
    }
    assert full.pop("output") != selected.pop("output")
    for key in ("input", "manifest", "repeats", "subset_counts"):
        selected.pop(key)
    assert full == selected


def row(identity, original, transformed):
    return {
        "id": identity,
        "original_trajectory_sha256": original,
        "trajectory_sha256": transformed,
    }


def test_exclusions_include_transitive_trajectory_lineage():
    full = [
        row("used", "a", "b"),
        row("other_id", "b", "c"),
        row("transitive", "c", "d"),
        row("heldout", "e", "f"),
    ]
    assert heldout_ids(list(reversed(full)), [full[0]]) == ["heldout"]


def test_confirmation_rejects_duplicate_ids_and_missing_lineage():
    example = row("same", "a", "b")
    with pytest.raises(ValueError, match="Duplicate"):
        heldout_ids([example, example], [example])
    with pytest.raises(KeyError):
        heldout_ids([{"id": "missing"}], [])


def test_report_recomputes_exclusions_and_preserves_quality_failure(
    tmp_path, monkeypatch
):
    full = [
        dict(
            row(str(i), str(i), str(i)),
            source="test",
            label=i % 2,
            prompt_sha256=str(i),
            tokens=10,
        )
        for i in range(5)
    ]
    inputs = {
        name: tmp_path / f"{name}.json"
        for name in ("input", "development", "capture", "reference")
    }
    inputs["input"].write_text("\n".join(json.dumps(r) for r in full))
    inputs["development"].write_text(json.dumps(full[0]))
    inputs["capture"].write_text('{"rows": []}')
    inputs["reference"].write_text('{"ids": ["0"]}')
    manifest = {
        "analysis_sha256": sha256_file(confirmation.Path(confirmation.__file__)),
        "source_files": {
            k: {"path": str(p), "sha256": sha256_file(p)} for k, p in inputs.items()
        },
        "heldout_ids": [str(i) for i in range(1, 5)],
        "heldout_rows": 4,
    }
    path = tmp_path / "exclusions.json"
    path.write_text(json.dumps(manifest))
    a, b = tmp_path / "a", tmp_path / "b"
    for root, amplitude in ((a, 0.1), (b, 0.2)):
        root.mkdir()
        predictions = [
            dict(r, score=1 - amplitude if r["label"] else amplitude) for r in full
        ]
        (root / "predictions_0.json").write_text(json.dumps(predictions))
    monkeypatch.setattr(
        confirmation,
        "compare_runs",
        lambda *args: {
            "runner_audit": {"changed": False},
            "candidate_serving_parity_passed": True,
        },
    )
    report = confirmation.report(a, b, path)
    assert report["heldout_rows"] == 4
    assert report["heldout_quality_gate"]["macro_brier_increase"] == pytest.approx(0.03)
    assert not report["confirmation_gates_passed"]
    assert report["heldout_drift_by_source"]["test"][
        "max_absolute_error"
    ] == pytest.approx(0.1)
    manifest["analysis_sha256"] = "wrong"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="analysis changed"):
        confirmation.report(a, b, path)
    manifest["analysis_sha256"] = sha256_file(confirmation.Path(confirmation.__file__))

    config = tmp_path / "selected.json"
    config.write_text('{"engine": {}, "output": "chosen"}')
    manifest["source_files"]["candidate_config"] = {
        "path": str(config),
        "sha256": sha256_file(config),
    }
    (b / "launch_config.json").write_text('{"engine": {}, "output": "changed"}')
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="frozen selection"):
        confirmation.report(a, b, path)
    (b / "launch_config.json").write_text(config.read_text())
    assert not confirmation.report(a, b, path)["confirmation_gates_passed"]
    manifest["heldout_ids"].append("0")
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="trajectory exclusion mismatch"):
        confirmation.report(a, b, path)
