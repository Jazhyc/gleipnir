"""Strict paired ID lineage, ordering and batch-shape repeat diagnostics."""

import hashlib

import pytest

from experiments.b200_optimized_id.analyze import compare
from experiments.b200_optimized_id.prepare import align_inputs


def test_input_alignment_rejects_label_text_source_and_order_drift():
    digest = hashlib.sha256(b"frozen input").hexdigest()
    rows = [
        {
            "id": "item",
            "prompt": "frozen input",
            "metadata": {
                "ground_truth": 1,
                "source_dataset": "source",
                "rendered_prompt_sha256": digest,
            },
        }
    ]
    control = [
        {"id": "item", "label": 1, "source": "source", "source_prompt_sha256": digest}
    ]
    align_inputs(rows, control)
    control[0]["label"] = 0
    with pytest.raises(ValueError, match="source/prompt/label/order"):
        align_inputs(rows, control)
    control[0]["label"] = 1
    rows[0]["prompt"] = "changed input"
    with pytest.raises(ValueError, match="source/prompt/label/order"):
        align_inputs(rows, control)


def population():
    rows = [
        {"id": str(i), "prompt_sha256": str(i), "dataset": "source", "label": i // 2}
        for i in range(4)
    ]
    baseline = [
        r | {"score": s, "margin": 2 * s - 1}
        for r, s in zip(rows, [0.1, 0.2, 0.8, 0.9], strict=True)
    ]
    return rows, baseline


def test_repeat_median_drift_and_instability_are_reported_separately():
    rows, baseline = population()
    runs = [
        [
            r | {"score": s, "margin": 2 * s - 1}
            for r, s in zip(rows, scores, strict=True)
        ]
        for scores in ([0.1, 0.4, 0.8, 0.9], [0.1, 0.6, 0.8, 0.9], [0.1, 0.6, 0.8, 0.9])
    ]
    result = compare(rows, baseline, runs)
    assert result["paired"]["threshold_flips"] == 1
    assert result["repeat_variation"]["threshold_unstable_ids"] == ["1"]
    assert result["score_drift"]["pooled"][
        "mean_absolute_score_difference"
    ] == pytest.approx(0.1)
    assert result["metric_deltas"]["macro"]["auroc"] == 0


def test_paired_analysis_rejects_reordering():
    rows, baseline = population()
    with pytest.raises(ValueError, match="identity drift"):
        compare(rows, baseline, [list(reversed(baseline))])


def test_metric_labels_cannot_drift_independently_of_prompt_hashes():
    rows, baseline = population()
    rows[0]["label"] = 1
    with pytest.raises(ValueError, match="source/label identity"):
        compare(rows, baseline, [baseline])


def test_single_pass_does_not_claim_repeat_variation():
    rows, baseline = population()
    result = compare(rows, baseline, [baseline])
    assert not result["repeat_variation"]["measured"]
    assert result["repeat_variation"]["passes"] == 1
    assert result["paired"]["threshold_flips"] == 0


def test_prepared_server_requires_exact_live_command(tmp_path, monkeypatch):
    from experiments.b200_optimized_id import run

    proc = tmp_path / "123"
    proc.mkdir()
    (proc / "cmdline").write_bytes(b"python\0-m\0expected.server\0")
    (proc / "stat").write_text("123 (python) S 1 2 3\n")
    monkeypatch.setattr(run, "Path", lambda p: tmp_path / str(p).removeprefix("/proc/"))
    process = run.ExistingServer(
        {"pid": 123, "command": ["python", "-m", "expected.server"]}
    )
    assert process.poll() is None
    (proc / "stat").write_text("123 (python) Z 1 2 3\n")
    assert process.poll() == -1
    (proc / "stat").write_text("123 (python) S 1 2 3\n")
    (proc / "cmdline").write_bytes(b"python\0-m\0other.server\0")
    with pytest.raises(ValueError, match="no longer live"):
        run.ExistingServer({"pid": 123, "command": ["python", "-m", "expected.server"]})
