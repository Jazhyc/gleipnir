"""Compatibility checks for the shared source package migration."""

import importlib
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

from gleipnir._compat import MODULE_ALIASES, install_aliases


def test_legacy_imports_share_state_and_preserve_canonical_metadata(monkeypatch):
    for legacy, canonical in MODULE_ALIASES.items():
        old = importlib.import_module(legacy)
        new = importlib.import_module(canonical)
        assert old is new
        assert new.__name__ == canonical
        assert new.__spec__.name == canonical
        sentinel = object()
        monkeypatch.setattr(old, "_layout_probe", sentinel, raising=False)
        assert new._layout_probe is sentinel


def test_aliases_require_no_compatibility_files():
    root = Path(__file__).resolve().parents[1] / "src"
    for legacy in MODULE_ALIASES:
        assert not (root / (legacy.replace(".", "/") + ".py")).exists()


def test_alias_registration_is_idempotent_and_reload_uses_the_implementation():
    original_finders = tuple(sys.meta_path)
    install_aliases()
    assert tuple(sys.meta_path) == original_finders
    old = importlib.import_module("gleipnir.prefix_cache")
    new = importlib.import_module("gleipnir.teachers.prefix_cache")
    assert importlib.reload(old) is new
    assert new.__spec__.name == "gleipnir.teachers.prefix_cache"


@pytest.mark.parametrize("canonical_first", [False, True])
def test_aliases_work_in_both_cold_import_orders(canonical_first):
    code = f"""
import importlib
aliases = {list(MODULE_ALIASES.items())!r}
for legacy, canonical in aliases:
    names = [legacy, canonical]
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


@pytest.mark.parametrize(
    "legacy,canonical",
    [
        ("gleipnir.openrouter_cli", "gleipnir.teachers.openrouter_cli"),
        ("gleipnir.monitoring_systems_screen", "gleipnir.campaigns.systems_screen"),
    ],
)
def test_legacy_and_canonical_command_help_match(legacy, canonical):
    outputs = []
    for name in (legacy, canonical):
        result = subprocess.run(
            [sys.executable, "-m", name, "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        assert result.stderr == ""
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1]


def test_repository_and_scratch_paths_survive_package_depth_change():
    from gleipnir.campaigns.systems_screen import ROOT
    from gleipnir.training.artifacts import systems_scratch

    repository = Path(__file__).resolve().parents[1]
    assert ROOT == repository
    assert systems_scratch() == repository / "results/systems_training_scratch"
