"""Transport validation without SSH, API calls, or paid infrastructure."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import runpod_cloud
from scripts.runpod_cloud import relative_path, ssh_argv


@pytest.mark.parametrize(
    "path",
    [
        "../.env",
        "/tmp/input",
        ".env",
        ".env.backup",
        ".runpod/id_ed25519",
        ".aws/credentials",
    ],
)
def test_transfer_refuses_credentials_and_escaping_paths(path: str) -> None:
    with pytest.raises(ValueError):
        relative_path(path)


def test_transfer_accepts_artifacts_and_public_env_template() -> None:
    assert relative_path("results/experiment/") == "results/experiment"
    assert relative_path(".env.example") == ".env.example"


def test_ssh_requires_live_direct_mapping() -> None:
    with pytest.raises(ValueError, match="not ready"):
        ssh_argv({"ssh": {"direct": None}}, Path("key"))


def test_ssh_preserves_port_and_uses_argument_vector() -> None:
    pod = {
        "ssh": {"direct": {"host": "198.51.100.2", "port": 12345, "username": "root"}}
    }
    argv = ssh_argv(pod, Path("key with spaces"))
    assert argv[argv.index("-p") + 1] == "12345"
    assert argv[argv.index("-i") + 1].endswith("key with spaces")
    assert argv[-1] == "root@198.51.100.2"
    assert "StrictHostKeyChecking=accept-new" in argv


@pytest.mark.parametrize(
    "direct",
    [
        {"host": "198.51.100.2;cat .env", "port": 12345, "username": "root"},
        {"host": "198.51.100.2", "port": 0, "username": "root"},
        {"host": "198.51.100.2", "port": 12345, "username": "root;cat .env"},
    ],
)
def test_ssh_rejects_invalid_connection_fields(direct: dict) -> None:
    with pytest.raises(ValueError):
        ssh_argv({"ssh": {"direct": direct}}, Path("key"))


def run_cli(monkeypatch, root: Path, action: str, paths: list[str]) -> None:
    pod_file = root / "pod.json"
    pod_file.write_text(
        json.dumps(
            {
                "ssh": {
                    "direct": {
                        "host": "198.51.100.2",
                        "port": 12345,
                        "username": "root",
                    }
                }
            }
        )
    )
    monkeypatch.setattr(runpod_cloud, "ROOT", root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runpod_cloud.py",
            "--pod-file",
            str(pod_file),
            "--key",
            str(root / "key"),
            action,
            "--",
            *paths,
        ],
    )
    runpod_cloud.main()


@pytest.mark.parametrize("action", ["push", "pull"])
def test_batch_transfers_preserve_paths_and_names_in_one_session(
    action, rsync_peer, monkeypatch
):
    source, destination = (
        (rsync_peer.local, rsync_peer.remote)
        if action == "push"
        else (rsync_peer.remote, rsync_peer.local)
    )
    files = [
        "first/same.json",
        "second/same.json",
        "#leading.txt",
        ";leading.txt",
        "-leading.txt",
        "space $name [1].json",
        "line\nbreak.txt",
        "unicode-é.txt",
        "bundle/deep/nested.json",
    ]
    for index, name in enumerate(files):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(index))
    (source / "bundle/incomplete.tmp").write_text("partial")
    (source / "unselected.txt").write_text("leave behind")
    paths = [*files[:-1], "bundle/", files[0]]
    run_cli(monkeypatch, rsync_peer.local, action, paths)

    for index, name in enumerate(files):
        assert (destination / name).read_text() == str(index)
    assert not (destination / "unselected.txt").exists()
    assert (destination / "bundle/incomplete.tmp").exists() == (action == "push")
    transfers = [
        (argv, kwargs) for argv, kwargs in rsync_peer.calls if argv[0] == "rsync"
    ]
    assert len(transfers) == 1
    assert len(rsync_peer.calls) == (2 if action == "push" else 1)
    assert transfers[0][1]["input"].count(b"./first/same.json\0") == 1


@pytest.mark.parametrize("action", ["push", "pull"])
@pytest.mark.parametrize("directory", [False, True])
def test_single_path_transfer_keeps_existing_destinations(
    action, directory, rsync_peer, monkeypatch
):
    source, destination = (
        (rsync_peer.local, rsync_peer.remote)
        if action == "push"
        else (rsync_peer.remote, rsync_peer.local)
    )
    name = "nested/folder/file.txt" if directory else "nested/file.txt"
    (source / name).parent.mkdir(parents=True, exist_ok=True)
    (source / name).write_text("original behavior")
    path = "nested/folder/" if directory else name
    run_cli(monkeypatch, rsync_peer.local, action, [path])
    assert (destination / name).read_text() == "original behavior"


@pytest.mark.parametrize("action", ["push", "pull"])
@pytest.mark.parametrize("bad_path", [".env", "../outside", "/outside", "bad\0name"])
def test_batch_validates_all_paths_before_remote_operations(
    action, bad_path, rsync_peer, monkeypatch
):
    (rsync_peer.local / "valid.txt").write_text("valid")
    with pytest.raises(ValueError):
        run_cli(monkeypatch, rsync_peer.local, action, ["valid.txt", bad_path])
    assert not rsync_peer.calls


def test_batch_missing_upload_fails_before_remote_operations(rsync_peer, monkeypatch):
    (rsync_peer.local / "valid.txt").write_text("valid")
    with pytest.raises(ValueError, match="does not exist"):
        run_cli(monkeypatch, rsync_peer.local, "push", ["valid.txt", "missing.txt"])
    assert not rsync_peer.calls


def test_batch_propagates_transfer_failure(rsync_peer, monkeypatch):
    def fail(argv, **kwargs):
        raise subprocess.CalledProcessError(23, argv)

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError) as failure:
        run_cli(monkeypatch, rsync_peer.local, "pull", ["a.txt", "b.txt"])
    assert failure.value.returncode == 23


@pytest.mark.parametrize("action", ["push", "pull"])
def test_batch_rejects_symlink_escape_before_remote_operations(
    action, rsync_peer, monkeypatch
):
    (rsync_peer.local / "valid.txt").write_text("valid")
    (rsync_peer.local / "outside").symlink_to(
        rsync_peer.remote, target_is_directory=True
    )
    with pytest.raises(ValueError, match="outside the repository"):
        run_cli(
            monkeypatch, rsync_peer.local, action, ["valid.txt", "outside/file.txt"]
        )
    assert not rsync_peer.calls
