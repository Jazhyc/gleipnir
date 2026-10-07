"""Compatibility checks for the shared source package migration."""

import importlib
import pickle
import subprocess
import sys

import pytest

ALIASES = [
    ("qwen35_loftq", "training.qwen35_loftq"),
    ("openrouter", "teachers.openrouter"),
    ("openrouter_cli", "teachers.openrouter_cli"),
    ("prefix_cache", "teachers.prefix_cache"),
    ("prefix_audit", "teachers.prefix_audit"),
    ("judge_injection", "data.judge_injection"),
    ("prefix_boundaries", "data.prefix_boundaries"),
    ("prefix_sampling", "data.prefix_sampling"),
    ("campaign_status", "campaigns.status"),
    ("binary_evaluation", "evaluation.binary"),
    ("metrics", "evaluation.metrics"),
    ("calibration", "evaluation.calibration"),
    ("decision_surface", "evaluation.decision_surface"),
    ("evaluation_lanes", "evaluation.lanes"),
    ("evaluation_shards", "evaluation.shards"),
    ("evaluation_watchdog", "evaluation.watchdog"),
    ("judge_injection_metrics", "evaluation.preferences"),
    ("monitoring_scoring", "evaluation.scoring"),
    ("monitoring_campaign_evaluation", "evaluation.campaign"),
]


@pytest.mark.parametrize("legacy,canonical", ALIASES)
def test_legacy_import_shares_module_state(legacy, canonical, monkeypatch):
    old = importlib.import_module(f"gleipnir.{legacy}")
    new = importlib.import_module(f"gleipnir.{canonical}")
    assert old is new
    sentinel = object()
    monkeypatch.setattr(old, "_layout_probe", sentinel, raising=False)
    assert new._layout_probe is sentinel


@pytest.mark.parametrize("canonical_first", [False, True])
def test_aliases_work_in_both_cold_import_orders(canonical_first):
    code = f"""
import importlib
aliases = {ALIASES!r}
for legacy, canonical in aliases:
    names = [f'gleipnir.{{legacy}}', f'gleipnir.{{canonical}}']
    if {canonical_first!r}:
        names.reverse()
    first, second = [importlib.import_module(name) for name in names]
    assert first is second, names
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)


def test_training_package_is_lazy():
    code = """
import sys
import gleipnir.training
import gleipnir.training.qwen35_loftq
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)


def test_training_exports_and_historical_pickle():
    from gleipnir import training
    from gleipnir.training import optimizers

    for name in training.__all__:
        assert getattr(training, name) is getattr(optimizers, name)
    assert pickle.loads(b"cgleipnir.training\nMuonAdamW\n.") is optimizers.MuonAdamW
    with pytest.raises(AttributeError):
        _ = training.missing_attribute


def test_legacy_and_canonical_annotation_cli_help_match():
    outputs = []
    for name in ("gleipnir.openrouter_cli", "gleipnir.teachers.openrouter_cli"):
        result = subprocess.run(
            [sys.executable, "-m", name, "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        assert result.stderr == ""
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1]
