"""Selection and numerical gates fail closed and do not depend on source order."""

import json
import subprocess
from collections import Counter
from pathlib import Path

import pytest

from experiments.local_inference import run
from experiments.local_inference.compare_runs import compare_runs
from experiments.local_inference.core import compare, parity_passes, select_subset


def test_comparison_checks_identity_and_paired_drift(tmp_path):
    baseline, candidate = tmp_path / "a", tmp_path / "b"
    result = dict.fromkeys(
        (
            "subset_sha256",
            "merge_manifest_sha256",
            "reference_sha256",
            "software",
            "runner_sha256",
        ),
        "same",
    )
    result.update(
        rows=2,
        prompt_tokens=20,
        repeats=[{}],
        median_seconds=2,
        median_prompt_tokens_per_second=10,
        metrics={},
    )
    rows = [
        {
            "id": str(i),
            "source": "test",
            "label": i,
            "prompt_sha256": str(i),
            "tokens": 10,
            "score": score,
            "logit_margin": score,
        }
        for i, score in enumerate((0.49, 0.9))
    ]
    for root in (baseline, candidate):
        root.mkdir()
        (root / "result.json").write_text(json.dumps(result))
        (root / "process_timing.json").write_text('{"seconds": 3}')
        (root / "predictions_0.json").write_text(json.dumps(rows))
    rows[0]["score"] = 0.51
    (candidate / "predictions_0.json").write_text(json.dumps(rows))
    report = compare_runs(baseline, candidate)
    assert report["score_drift"]["threshold_flips"] == 1
    assert report["scoring_speedup"] == 1
    rows[0]["prompt_sha256"] = "changed"
    (candidate / "predictions_0.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="prediction"):
        compare_runs(baseline, candidate)


def test_prefill_candidate_changes_only_budget_and_output():
    root = Path(__file__).resolve().parents[1] / "experiments/local_inference"
    baseline = json.loads((root / "iteration32.json").read_text())
    candidate = json.loads((root / "prefill4096.json").read_text())
    assert candidate.pop("output") != baseline.pop("output")
    assert candidate["engine"]["max_num_batched_tokens"] == 4096
    candidate["engine"]["max_num_batched_tokens"] = 2048
    assert candidate == baseline


@pytest.mark.parametrize(
    ("filename", "scheme"),
    [("fp8.json", "fp8_per_tensor"), ("fp8_channel.json", "fp8_per_channel")],
)
def test_fp8_candidate_changes_only_quantization_and_output(filename, scheme):
    root = Path(__file__).resolve().parents[1] / "experiments/local_inference"
    baseline = json.loads((root / "iteration32.json").read_text())
    candidate = json.loads((root / filename).read_text())
    assert candidate.pop("output") != baseline.pop("output")
    assert candidate["engine"].pop("quantization") == scheme
    assert candidate == baseline


def test_custom_config_only_runs_benchmark(tmp_path, monkeypatch):
    output = tmp_path / "candidate"
    config = tmp_path / "candidate.json"
    config.write_text(json.dumps({"output": str(output)}))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run, "ROOT", tmp_path / "results")
    monkeypatch.setattr(run.sys, "argv", ["run", "--config", str(config)])
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(run.subprocess, "run", fake_run)
    run.main()
    assert len(calls) == 1
    assert calls[0][-3:] == [
        "experiments.local_inference.benchmark",
        "--config",
        str(config),
    ]
    assert json.loads((output / "process_timing.json").read_text())["returncode"] == 0


def test_selection_preserves_quotas_lengths_and_order_independence():
    rows = [
        {"id": f"{source}:{label}:{i}", "source": source, "label": label, "tokens": i}
        for source in ("stride", "gloom")
        for label in (0, 1)
        for i in range(100)
    ]
    quotas = {f"{s}:{label}": 13 for s in ("stride", "gloom") for label in (0, 1)}
    selected = select_subset(rows, quotas, 42)
    assert selected == select_subset(list(reversed(rows)), quotas, 42)
    assert len({r["id"] for r in selected}) == 52
    assert Counter(f"{r['source']}:{r['label']}" for r in selected) == quotas
    assert Counter(r["length_quartile"] for r in selected) == {
        0: 16,
        1: 12,
        2: 12,
        3: 12,
    }
    assert selected != select_subset(rows, quotas, 43)
    with pytest.raises(ValueError, match="duplicate"):
        select_subset(rows + rows[:1], quotas, 42)


def test_parity_checks_include_threshold_boundary_and_nonfinite():
    report = compare([0.2, 0.49, 0.9], [0.2, 0.5, 0.9])
    assert report["threshold_flips"] == 1
    limits = {
        "mean_absolute_error": 0.02,
        "max_absolute_error": 0.1,
        "min_correlation": 0.99,
    }
    assert parity_passes(report, limits)
    assert not parity_passes(compare([0.2, 0.9], [0.8, 0.3]), limits)
    with pytest.raises(ValueError, match="nonfinite"):
        compare([0.2], [float("nan")])
    with pytest.raises(ValueError, match="coverage"):
        compare([0.2], [0.2, 0.3])


def test_small_screen_retains_every_source_label_length_bin():
    rows = [
        {"id": f"{s}:{label}:{i}", "source": s, "label": label, "tokens": i}
        for s in ("test_stride", "gloom_exfiltration")
        for label in (0, 1)
        for i in range(100)
    ]
    counts = {
        "test_stride:0": 4,
        "test_stride:1": 6,
        "gloom_exfiltration:0": 11,
        "gloom_exfiltration:1": 11,
    }
    selected = select_subset(rows, counts, 20260923)
    assert len(selected) == 32
    assert Counter(f"{r['source']}:{r['label']}" for r in selected) == counts
    assert (
        len({(r["source"], r["label"], r["length_quartile"]) for r in selected}) == 16
    )


def test_existing_benchmark_timing_is_not_overwritten(tmp_path, monkeypatch):
    output = tmp_path / "baseline"
    output.mkdir()
    timing = output / "process_timing.json"
    timing.write_text('{"seconds": 85.3}\n')
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"output": str(output)}))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run, "CONFIG", config)
    monkeypatch.setattr(run, "ROOT", tmp_path / "results")
    monkeypatch.setattr(run.sys, "argv", ["run"])
    monkeypatch.setenv("PATH", run.os.environ["PATH"])
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(run.subprocess, "run", fake_run)
    with pytest.raises(FileExistsError, match="Preserve"):
        run.main()
    assert len(calls) == 3
    assert timing.read_text() == '{"seconds": 85.3}\n'
