"""Exercise cloud file transfers using a local rsync peer instead of SSH."""

import shlex
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def rsync_peer(tmp_path, monkeypatch):
    local, remote = tmp_path / "local", tmp_path / "remote"
    local.mkdir()
    remote.mkdir()
    calls = []
    real_run = subprocess.run
    prefixes = (
        "root@198.51.100.2:/workspace/gleipnir",
        "ubuntu@192.0.2.1:gleipnir",
    )

    def run(argv, **kwargs):
        calls.append((list(argv), kwargs.copy()))
        if argv[0] == "ssh":
            host = next(
                a for a in argv if a in {"root@198.51.100.2", "ubuntu@192.0.2.1"}
            )
            command = argv[argv.index(host) + 1 :]
            words = shlex.split(command[0]) if len(command) == 1 else command
            assert words[:2] == ["mkdir", "-p"]
            for path in words[2:]:
                prefix = (
                    "/workspace/gleipnir" if host.startswith("root@") else "gleipnir"
                )
                assert path.startswith(prefix)
                Path(str(remote) + path.removeprefix(prefix)).mkdir(
                    parents=True, exist_ok=True
                )
            return subprocess.CompletedProcess(argv, 0)
        assert argv[0] == "rsync"
        if shutil.which("rsync") is None:
            pytest.skip("local transfer verification requires rsync")
        mapped = list(argv)
        for index in (-2, -1):
            for prefix in prefixes:
                if mapped[index].startswith(prefix):
                    mapped[index] = str(remote) + mapped[index].removeprefix(prefix)
                    break
            else:
                assert mapped[index].startswith(str(local))
        return real_run(mapped, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    return SimpleNamespace(local=local, remote=remote, calls=calls)
