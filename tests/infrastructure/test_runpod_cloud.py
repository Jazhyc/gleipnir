"""Transport validation without SSH, API calls, or paid infrastructure."""

from pathlib import Path

import pytest

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
