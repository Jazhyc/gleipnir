"""Validate relocated live sources without changing archived evidence."""

import ast
import hashlib
import subprocess
from pathlib import Path

import pytest

from experiments.monitoring_branching.train import validate_smoke_sources
from gleipnir._compat import canonical_source_reference


@pytest.mark.parametrize(
    "source,expected",
    [
        ("src/gleipnir/packed_training.py", "src/gleipnir/training/packed.py"),
        ("src/gleipnir/branch_data.py", "src/gleipnir/data/branches.py"),
        (
            "src/gleipnir/monitoring_systems_screen.py",
            "src/gleipnir/campaigns/systems_screen.py",
        ),
        ("src/gleipnir/native_fp4_training.py", "src/gleipnir/native_fp4_training.py"),
        (
            "results/prior/executed_sources/src/gleipnir/packed_training.py",
            "results/prior/executed_sources/src/gleipnir/packed_training.py",
        ),
        ("data/source.jsonl", "data/source.jsonl"),
    ],
)
def test_only_live_moved_source_references_are_translated(source, expected):
    assert canonical_source_reference(source) == expected


@pytest.mark.parametrize("canonical_keys", [False, True])
def test_smoke_source_reuse_still_requires_exact_recorded_bytes(
    tmp_path, monkeypatch, canonical_keys
):
    monkeypatch.chdir(tmp_path)
    files = {}
    for name in (
        "branch_model.py",
        "branch_training.py",
        "branch_data.py",
        "prefix_loss.py",
    ):
        old = "src/gleipnir/" + name
        current = canonical_source_reference(old)
        path = Path(current)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("recorded " + name).encode())
        files[current if canonical_keys else old] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    validate_smoke_sources(files)
    conflicting = {
        **files,
        "src/gleipnir/prefix_loss.py": "old checksum",
        canonical_source_reference("src/gleipnir/prefix_loss.py"): "new checksum",
    }
    with pytest.raises(ValueError, match="numerical path changed"):
        validate_smoke_sources(conflicting)
    Path(canonical_source_reference("src/gleipnir/prefix_loss.py")).write_bytes(
        b"changed objective"
    )
    with pytest.raises(ValueError, match="numerical path changed"):
        validate_smoke_sources(files)
    with pytest.raises(ValueError, match="numerical path changed"):
        validate_smoke_sources({})


def test_launcher_source_literals_identify_existing_implementations():
    root = Path(__file__).resolve().parents[1]
    tracked = (
        subprocess.check_output(
            ["git", "-C", str(root), "ls-files", "-z", "--", "experiments"],
        )
        .decode()
        .split("\0")
    )
    for relative in tracked:
        path = root / relative
        if path.suffix != ".py" or path.name.startswith("test_"):
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value.startswith("src/gleipnir/")
                and node.value.endswith(".py")
            ):
                assert any(p.is_file() for p in root.glob(node.value)), (
                    path,
                    node.value,
                )
