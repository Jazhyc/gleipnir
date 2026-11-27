"""Repository-root and experiment-directory lookup."""

import subprocess
import sys
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
    assert experiment_dir("cascade") == ROOT / "experiments/evaluation/cascade"


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
        ROOT / "experiments/training/tool_trajectory_monitoring/prompts"
    )
    assert prompt_contract.load_prompt_set().teacher.template_sha256


def _experiment_tree():
    from gleipnir._compat import EXPERIMENT_CATEGORIES

    base = ROOT / "experiments"
    top = {p.name for p in base.iterdir() if p.is_dir() and p.name != "__pycache__"}
    return base, top, EXPERIMENT_CATEGORIES


def test_experiment_registry_matches_the_tree():
    from gleipnir._compat import EXPERIMENT_MOVES

    base, top, categories = _experiment_tree()
    for category, names in categories.items():
        assert category in top
        on_disk = {
            p.name
            for p in (base / category).iterdir()
            if p.is_dir() and p.name != "__pycache__"
        }
        # New experiments start in a category; the registry lists moved ones.
        assert set(names) <= on_disk, category
    kept = top - set(categories)
    assert not kept & set(EXPERIMENT_MOVES)
    names = [n for group in categories.values() for n in group] + sorted(kept)
    assert len(names) == len(set(names)), "experiment names must stay unique"


def test_category_indexes_list_every_experiment():
    base, top, categories = _experiment_tree()
    index = (base / "README.md").read_text()
    for category in categories:
        assert f"({category}/)" in index
        listing = (base / category / "README.md").read_text()
        for path in (base / category).iterdir():
            if path.is_dir() and path.name != "__pycache__":
                assert f"[`{path.name}`]({path.name}/)" in listing, path
    for name in top - set(categories):
        assert f"({name}/)" in index, name


def test_flat_experiment_names_keep_resolving_in_a_cold_process():
    code = """
import importlib
old = importlib.import_module("experiments.cascade.cascade")
new = importlib.import_module("experiments.evaluation.cascade.cascade")
assert old is new, (old, new)
spaced = importlib.import_module("experiments.student_injection_awareness.prepare")
assert spaced.__file__.endswith(
    "experiments/injection/student_injection_awareness/prepare.py"
), spaced.__file__
"""
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.student_injection_awareness.summarize",
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert "--size" in result.stdout


def test_recorded_flat_experiment_paths_translate():
    from gleipnir._compat import canonical_source_reference

    assert canonical_source_reference("experiments/cascade/cascade.py") == (
        "experiments/evaluation/cascade/cascade.py"
    )
    kept = "experiments/b200_vllm031/validate.py"
    assert canonical_source_reference(kept) == kept
    assert canonical_source_reference("results/cascade/x.json") == (
        "results/cascade/x.json"
    )
