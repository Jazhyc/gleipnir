"""Owned merge replacement and BF16-only launch guards."""

import json
from pathlib import Path

import pytest

from experiments.filtered_fp4_ood import launch, run


def test_restore_rejects_unowned_slot(tmp_path):
    with pytest.raises(ValueError, match="unsafe"):
        run.restore_merge(
            tmp_path, {"teacher_merge": str(tmp_path / "other")}, tmp_path
        )


def test_restore_rejects_changed_predecessor_without_deletion(monkeypatch, tmp_path):
    config = {
        "teacher_merge": "/tmp/gleipnir-merged/filtered-fp4-fa4-baseline",
        "merge_predecessor": "previous.json",
    }
    monkeypatch.setattr(Path, "is_symlink", lambda self: False)
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda self: json.dumps(
            {
                "adapter_sha256": "old"
                if self.name == "merge_manifest.json"
                else "expected"
            }
        ),
    )
    monkeypatch.setattr(run.shutil, "rmtree", lambda *a: pytest.fail("must not delete"))
    with pytest.raises(ValueError, match="ownership"):
        run.restore_merge(tmp_path, config, tmp_path)


def test_launch_refuses_active_gpu(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "experiments/filtered_fp4_ood/config.yaml"
    path.parent.mkdir(parents=True)
    path.write_text("experiment: test\n")
    monkeypatch.setattr(launch.subprocess, "check_output", lambda *a, **k: "123\n")
    with pytest.raises(RuntimeError, match="active compute"):
        launch.main()
