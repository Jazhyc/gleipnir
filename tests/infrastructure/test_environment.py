import os
import sys
from pathlib import Path

import pytest

from gleipnir._environment import expose_interpreter_scripts


def test_interpreter_scripts_lead_path_once(monkeypatch: pytest.MonkeyPatch) -> None:
    scripts = str(Path(sys.executable).parent.absolute())
    monkeypatch.setenv("PATH", os.pathsep.join(["/usr/bin", scripts, "/bin"]))
    expose_interpreter_scripts()
    entries = os.environ["PATH"].split(os.pathsep)
    assert entries == [scripts, "/usr/bin", "/bin"]
    expose_interpreter_scripts()
    assert os.environ["PATH"].split(os.pathsep) == entries


def test_scripts_directory_is_not_resolved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    real = tmp_path / "cache" / "python3.12"
    real.parent.mkdir()
    real.touch()
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").symlink_to(real)
    monkeypatch.setattr(sys, "executable", str(venv_bin / "python"))
    monkeypatch.setenv("PATH", "/usr/bin")
    expose_interpreter_scripts()
    assert os.environ["PATH"].split(os.pathsep)[0] == str(venv_bin)
