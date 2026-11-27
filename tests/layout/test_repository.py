"""Repository-root and experiment-directory lookup."""

from pathlib import Path

import pytest

from gleipnir.repository import experiment_dir, repo_root
from tests.helpers.paths import ROOT


def _checkout(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text("")
    (tmp_path / "experiments").mkdir()
    return tmp_path


def test_repo_root_walks_up_from_any_depth(tmp_path):
    root = _checkout(tmp_path)
    flat = root / "experiments/alpha/run.py"
    nested = root / "experiments/category/beta/sub/run.py"
    for path in (flat, nested):
        path.parent.mkdir(parents=True)
        path.write_text("")
        assert repo_root(path) == root
    assert repo_root(root) == root


def test_repo_root_requires_a_checkout(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")  # no experiments/ beside it
    with pytest.raises(RuntimeError, match="no Gleipnir checkout"):
        repo_root(tmp_path / "file.py")


def test_this_checkout_is_found():
    assert repo_root(__file__) == ROOT
    assert experiment_dir("cascade") == ROOT / "experiments/cascade"


def test_experiment_dir_resolves_flat_and_categorized(tmp_path):
    root = _checkout(tmp_path)
    (root / "experiments/flat_one").mkdir()
    (root / "experiments/serving/nested_one").mkdir(parents=True)
    assert experiment_dir("flat_one", root) == root / "experiments/flat_one"
    assert experiment_dir("nested_one", root) == (
        root / "experiments/serving/nested_one"
    )


def test_experiment_dir_rejects_missing_ambiguous_and_invalid(tmp_path):
    root = _checkout(tmp_path)
    (root / "experiments/a/twin").mkdir(parents=True)
    (root / "experiments/b/twin").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        experiment_dir("absent", root)
    with pytest.raises(RuntimeError, match="ambiguous"):
        experiment_dir("twin", root)
    for name in ("", "a/twin", "..", "__pycache__"):
        with pytest.raises(ValueError):
            experiment_dir(name, root)


def test_moved_prompt_contract_keeps_its_historical_import_and_source_path():
    import importlib

    from gleipnir._compat import canonical_source_reference
    from gleipnir.teachers import prompt_contract

    legacy = importlib.import_module("experiments.tool_trajectory_monitoring.prompting")
    assert legacy is prompt_contract
    assert canonical_source_reference(
        "experiments/tool_trajectory_monitoring/prompting.py"
    ) == ("src/gleipnir/teachers/prompt_contract.py")
    assert canonical_source_reference("src/gleipnir/metrics.py") == (
        "src/gleipnir/evaluation/metrics.py"
    )
    assert prompt_contract.PROMPT_DIRECTORY == (
        ROOT / "experiments/tool_trajectory_monitoring/prompts"
    )
    assert prompt_contract.load_prompt_set().teacher.template_sha256
