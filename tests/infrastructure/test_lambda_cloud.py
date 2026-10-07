from __future__ import annotations

import importlib.util
import json
import os
import shlex
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.helpers.paths import ROOT

MODULE_PATH = ROOT / "scripts/lambda_cloud.py"
SPEC = importlib.util.spec_from_file_location("lambda_cloud", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
lambda_cloud = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lambda_cloud)


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def test_client_sends_bearer_token_without_putting_it_in_url() -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["authorization"] = request.get_header("Authorization")
        captured["timeout"] = timeout
        return FakeResponse({"data": []})

    client = lambda_cloud.LambdaCloudClient(
        "top-secret", base_url="https://example.invalid/api/v1"
    )
    with patch.object(lambda_cloud.urllib.request, "urlopen", fake_urlopen):
        assert client.instances() == []

    assert captured == {
        "url": "https://example.invalid/api/v1/instances",
        "authorization": "Bearer top-secret",
        "timeout": 30.0,
    }
    assert "top-secret" not in captured["url"]


def test_launch_payload_uses_one_named_campaign_instance() -> None:
    response = FakeResponse({"data": {"instance_ids": ["instance-1"]}})
    requests = []

    def fake_urlopen(request, timeout):
        requests.append(request)
        return response

    client = lambda_cloud.LambdaCloudClient("key")
    with patch.object(lambda_cloud.urllib.request, "urlopen", fake_urlopen):
        ids = client.launch(
            {
                "region_name": "us-west-2",
                "instance_type_name": "gpu_1x_a100_sxm4",
                "ssh_key_names": ["gleipnir-habrok"],
                "file_system_names": [],
                "quantity": 1,
                "name": "gleipnir-prompt-campaign",
            }
        )

    assert ids == ["instance-1"]
    request = requests[0]
    assert request.method == "POST"
    assert json.loads(request.data) == {
        "region_name": "us-west-2",
        "instance_type_name": "gpu_1x_a100_sxm4",
        "ssh_key_names": ["gleipnir-habrok"],
        "file_system_names": [],
        "quantity": 1,
        "name": "gleipnir-prompt-campaign",
    }


def test_campaign_matching_is_exact_and_validated() -> None:
    instances = [
        {"name": "gleipnir-prompt-campaign", "id": "wanted"},
        {"name": "gleipnir-prompt-campaign-old", "id": "other"},
    ]
    assert lambda_cloud.matching_instances(instances, "prompt-campaign") == [
        {"name": "gleipnir-prompt-campaign", "id": "wanted"}
    ]
    with pytest.raises(lambda_cloud.LambdaCloudError, match="campaign must start"):
        lambda_cloud.campaign_instance_name("../bad")


def test_campaign_matching_accepts_exact_console_title_as_fallback() -> None:
    instances = [
        {"name": "Eleuther-Slayer", "id": "wanted"},
        {"name": "Eleuther-Slayer-old", "id": "other"},
    ]
    assert lambda_cloud.matching_instances(instances, "eleuther-slayer") == [
        {"name": "Eleuther-Slayer", "id": "wanted"}
    ]


def test_campaign_matching_prefers_managed_prefix_over_console_title() -> None:
    instances = [
        {"name": "gleipnir-eleuther-slayer", "id": "managed"},
        {"name": "Eleuther-Slayer", "id": "console"},
    ]
    assert lambda_cloud.matching_instances(instances, "eleuther-slayer") == [
        {"name": "gleipnir-eleuther-slayer", "id": "managed"}
    ]


def test_parser_requires_explicit_confirmation_for_launch() -> None:
    parser = lambda_cloud.build_parser()
    args = parser.parse_args(
        [
            "launch",
            "--campaign",
            "prompt-campaign",
            "--instance-type",
            "gpu_1x_a100_sxm4",
            "--region",
            "us-west-2",
        ]
    )
    assert args.yes is False
    assert args.allow_non_x86 is False


def test_sync_code_requires_explicit_uncommitted_opt_in() -> None:
    parser = lambda_cloud.build_parser()
    args = parser.parse_args(["sync-code", "--campaign", "prompt-campaign"])
    assert args.include_uncommitted is False
    committed = parser.parse_args(["sync-commit", "--campaign", "prompt-campaign"])
    assert committed.revision == "HEAD"


def test_remote_secret_payload_is_allowlisted_and_shell_quoted() -> None:
    with patch.dict(os.environ, {"HF_TOKEN": "token with spaces"}, clear=False):
        payload = lambda_cloud.build_remote_secret_payload(["HF_TOKEN"])
    assert "export HF_TOKEN='token with spaces'" in payload
    with pytest.raises(lambda_cloud.LambdaCloudError, match="non-allowlisted"):
        lambda_cloud.build_remote_secret_payload(["LAMBDA_API_KEY"])
    with pytest.raises(lambda_cloud.LambdaCloudError, match="non-allowlisted"):
        lambda_cloud.build_remote_secret_payload(["LAMBDA_API_KEY"])


def test_ssh_command_is_one_shell_quoted_remote_argument() -> None:
    args = SimpleNamespace(
        command=["--", "bash", "-lc", "printf '%s\\n' 'hello world'"],
    )
    with (
        patch.object(
            lambda_cloud,
            "active_ssh_target",
            return_value=(Path("/tmp/test-key"), {"ip": "192.0.2.1"}),
        ),
        patch.object(lambda_cloud, "run_checked") as run_checked,
    ):
        lambda_cloud.command_ssh(args, object())

    argv = run_checked.call_args.args[0]
    assert argv[-1] == """bash -lc 'printf '"'"'%s\\n'"'"' '"'"'hello world'"'"''"""
    assert "BatchMode=yes" in argv
    assert "ServerAliveInterval=15" in argv
    assert "ServerAliveCountMax=3" in argv
    assert "ControlMaster=auto" in argv
    assert "ControlPersist=600" in argv


def test_ssh_and_rsync_share_resilient_transport_options() -> None:
    private_key = Path("/tmp/test-key")
    ssh = lambda_cloud.ssh_argv(private_key, "192.0.2.1")
    rsync_ssh = shlex.split(lambda_cloud.rsync_ssh_command(private_key))

    for option in (
        "BatchMode=yes",
        "ConnectTimeout=10",
        "ConnectionAttempts=3",
        "ServerAliveInterval=15",
        "ServerAliveCountMax=3",
        "TCPKeepAlive=yes",
        "ProxyCommand=",
        "ControlMaster=auto",
        "ControlPersist=600",
    ):
        if option.endswith("="):
            assert any(value.startswith(option) for value in ssh)
            assert any(value.startswith(option) for value in rsync_ssh)
        else:
            assert option in ssh
            assert option in rsync_ssh
    assert any(value.startswith("ControlPath=") for value in ssh)
    assert any(value.startswith("ControlPath=") for value in rsync_ssh)
    proxy = next(value for value in ssh if value.startswith("ProxyCommand="))
    assert "tcp_mss_proxy.py" in proxy
    assert "--mss 1400" in proxy


def test_rsync_is_resumable_and_has_an_io_timeout() -> None:
    argv = lambda_cloud.rsync_argv(Path("/tmp/test-key"))
    assert "--partial" in argv
    assert "--partial-dir=.rsync-partial" in argv
    assert "--timeout=60" in argv


@pytest.mark.parametrize("input_data", [None, b"./file one\0./file\ntwo\0"])
def test_transfer_retries_with_bounded_backoff(input_data) -> None:
    results = [
        SimpleNamespace(returncode=30),
        SimpleNamespace(returncode=12),
        SimpleNamespace(returncode=0),
    ]
    with (
        patch.object(lambda_cloud.subprocess, "run", side_effect=results) as run,
        patch.object(lambda_cloud.time, "sleep") as sleep,
    ):
        lambda_cloud.run_transfer(["rsync", "source", "target"], input_data=input_data)

    assert run.call_count == 3
    assert all(call.kwargs["input"] == input_data for call in run.call_args_list)
    assert [call.args[0] for call in sleep.call_args_list] == [1, 2]


def test_status_reads_a_bounded_project_relative_json_file() -> None:
    args = SimpleNamespace(remote_path="results/example/status.json")
    with (
        patch.object(
            lambda_cloud,
            "active_ssh_target",
            return_value=(Path("/tmp/test-key"), {"ip": "192.0.2.1"}),
        ),
        patch.object(lambda_cloud, "run_checked") as run_checked,
    ):
        lambda_cloud.command_status(args, object())

    remote_command = run_checked.call_args.args[0][-1]
    assert "gleipnir/results/example/status.json" in remote_command
    assert str(lambda_cloud.MAX_STATUS_BYTES) in remote_command


def test_parser_exposes_bounded_status_reader() -> None:
    args = lambda_cloud.build_parser().parse_args(
        [
            "status",
            "--campaign",
            "monitor-foundation",
            "--remote-path",
            "results/example/status.json",
        ]
    )
    assert args.handler is lambda_cloud.command_status
    assert args.remote_path == "results/example/status.json"


def test_compute_probe_renders_project_environment_script() -> None:
    with (
        patch.object(
            lambda_cloud,
            "active_ssh_target",
            return_value=(Path("/tmp/test-key"), {"ip": "192.0.2.1"}),
        ),
        patch.object(lambda_cloud, "run_checked") as run_checked,
    ):
        lambda_cloud.command_compute_probe(SimpleNamespace(), object())

    script = run_checked.call_args.kwargs["input_text"]
    assert 'source "$HOME/.config/gleipnir/runtime.env"' in script
    assert 'cd "$HOME/gleipnir"' in script
    assert 'print(f"torch={torch.__version__}")' in script


def test_public_key_fingerprint_matches_openssh_shape() -> None:
    public_key = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIEqHxWB8sExampleOnly"
    assert lambda_cloud.public_key_fingerprint(public_key).startswith("SHA256:")


@pytest.mark.parametrize(
    "path",
    ["/etc/passwd", "../outside", "results/../../outside"],
)
def test_remote_path_rejects_absolute_or_parent_traversal(path: str) -> None:
    with pytest.raises(lambda_cloud.LambdaCloudError):
        lambda_cloud.ensure_relative_remote_path(path)


def test_safe_api_error_does_not_include_authorization_header() -> None:
    error = urllib.error.HTTPError(
        "https://example.invalid/api/v1/instances",
        401,
        "Unauthorized",
        {"Authorization": "Bearer top-secret"},
        None,
    )
    assert lambda_cloud.safe_api_error(error) == "Unauthorized"


@pytest.mark.parametrize("action", ["push", "pull"])
def test_batch_transfers_preserve_layout_in_one_session(
    action, rsync_peer, monkeypatch
):
    source, destination = (
        (rsync_peer.local, rsync_peer.remote)
        if action == "push"
        else (rsync_peer.remote, rsync_peer.local)
    )
    files = [
        "inputs/same.json",
        "results/same.json",
        "#leading.txt",
        "space $name [1].json",
        "line\nbreak.txt",
        "bundle/deep/file.txt",
    ]
    for index, name in enumerate(files):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(index))
    (source / "unselected.txt").write_text("leave behind")
    flag = "--local-path" if action == "push" else "--remote-path"
    args = lambda_cloud.build_parser().parse_args(
        [action, "--campaign", "example", flag, *files[:-1], "bundle/", files[0]]
    )
    monkeypatch.setattr(lambda_cloud, "ROOT", rsync_peer.local)
    with patch.object(
        lambda_cloud,
        "active_ssh_target",
        return_value=(Path("/tmp/key"), {"ip": "192.0.2.1"}),
    ) as target:
        args.handler(args, object())
    assert target.call_count == 1
    for index, name in enumerate(files):
        assert (destination / name).read_text() == str(index)
    assert not (destination / "unselected.txt").exists()
    assert len(rsync_peer.calls) == (2 if action == "push" else 1)
    transfers = [
        (argv, kwargs) for argv, kwargs in rsync_peer.calls if argv[0] == "rsync"
    ]
    assert len(transfers) == 1
    assert transfers[0][1]["input"].count(b"./inputs/same.json\0") == 1


@pytest.mark.parametrize("action", ["push", "pull"])
@pytest.mark.parametrize("directory", [False, True])
def test_single_path_rename_keeps_existing_behavior(
    action, directory, rsync_peer, monkeypatch
):
    source, destination = (
        (rsync_peer.local, rsync_peer.remote)
        if action == "push"
        else (rsync_peer.remote, rsync_peer.local)
    )
    source_name = "original/inside.txt" if directory else "original.txt"
    destination_name = "renamed/inside.txt" if directory else "renamed.txt"
    (source / source_name).parent.mkdir(parents=True, exist_ok=True)
    (source / source_name).write_text("original behavior")
    source_path = "original" if directory else "original.txt"
    destination_path = "renamed" if directory else "renamed.txt"
    local_path, remote_path = (
        (source_path, destination_path)
        if action == "push"
        else (destination_path, source_path)
    )
    args = lambda_cloud.build_parser().parse_args(
        [
            action,
            "--campaign",
            "example",
            "--local-path",
            local_path,
            "--remote-path",
            remote_path,
            *(["--directory"] if action == "pull" and directory else []),
        ]
    )
    monkeypatch.setattr(lambda_cloud, "ROOT", rsync_peer.local)
    with patch.object(
        lambda_cloud,
        "active_ssh_target",
        return_value=(Path("/tmp/key"), {"ip": "192.0.2.1"}),
    ):
        args.handler(args, object())
    assert (destination / destination_name).read_text() == "original behavior"


@pytest.mark.parametrize("action", ["push", "pull"])
@pytest.mark.parametrize("bad_path", ["../outside", "/outside", "bad\0name"])
def test_batch_validates_before_target_lookup(
    action, bad_path, rsync_peer, monkeypatch
):
    (rsync_peer.local / "valid.txt").write_text("valid")
    flag = "--local-path" if action == "push" else "--remote-path"
    args = lambda_cloud.build_parser().parse_args(
        [action, "--campaign", "example", flag, "valid.txt", bad_path]
    )
    monkeypatch.setattr(lambda_cloud, "ROOT", rsync_peer.local)
    with patch.object(lambda_cloud, "active_ssh_target") as target:
        with pytest.raises(lambda_cloud.LambdaCloudError):
            args.handler(args, object())
    target.assert_not_called()
    assert not rsync_peer.calls


@pytest.mark.parametrize("action", ["push", "pull"])
def test_batch_rejects_ambiguous_rename_before_target_lookup(action):
    source_flag = "--local-path" if action == "push" else "--remote-path"
    destination_flag = "--remote-path" if action == "push" else "--local-path"
    args = lambda_cloud.build_parser().parse_args(
        [
            action,
            "--campaign",
            "example",
            source_flag,
            "a",
            "b",
            destination_flag,
            "dest",
        ]
    )
    with patch.object(lambda_cloud, "active_ssh_target") as target:
        with pytest.raises(lambda_cloud.LambdaCloudError, match="single"):
            args.handler(args, object())
    target.assert_not_called()


@pytest.mark.parametrize("action", ["push", "pull"])
def test_repeated_source_option_keeps_every_path(action):
    flag = "--local-path" if action == "push" else "--remote-path"
    args = lambda_cloud.build_parser().parse_args(
        [action, "--campaign", "example", flag, "a", "b", flag, "c"]
    )
    assert getattr(args, flag[2:].replace("-", "_")) == ["a", "b", "c"]


def test_batch_missing_upload_fails_before_target_lookup(rsync_peer, monkeypatch):
    (rsync_peer.local / "valid.txt").write_text("valid")
    args = lambda_cloud.build_parser().parse_args(
        ["push", "--campaign", "example", "--local-path", "valid.txt", "missing.txt"]
    )
    monkeypatch.setattr(lambda_cloud, "ROOT", rsync_peer.local)
    with patch.object(lambda_cloud, "active_ssh_target") as target:
        with pytest.raises(lambda_cloud.LambdaCloudError, match="does not exist"):
            args.handler(args, object())
    target.assert_not_called()
    assert not rsync_peer.calls


@pytest.mark.parametrize("action", ["push", "pull"])
def test_batch_rejects_symlink_escape_before_target_lookup(
    action, rsync_peer, monkeypatch
):
    (rsync_peer.local / "valid.txt").write_text("valid")
    (rsync_peer.local / "outside").symlink_to(
        rsync_peer.remote, target_is_directory=True
    )
    flag = "--local-path" if action == "push" else "--remote-path"
    args = lambda_cloud.build_parser().parse_args(
        [action, "--campaign", "example", flag, "valid.txt", "outside/file.txt"]
    )
    monkeypatch.setattr(lambda_cloud, "ROOT", rsync_peer.local)
    with patch.object(lambda_cloud, "active_ssh_target") as target:
        with pytest.raises(
            lambda_cloud.LambdaCloudError, match="inside the repository"
        ):
            args.handler(args, object())
    target.assert_not_called()
    assert not rsync_peer.calls
