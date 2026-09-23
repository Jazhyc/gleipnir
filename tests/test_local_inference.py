"""Selection and numerical gates fail closed and do not depend on source order."""

import json
import subprocess
from collections import Counter

import pytest

from experiments.local_inference import run
from experiments.local_inference.core import compare, parity_passes, select_subset


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
