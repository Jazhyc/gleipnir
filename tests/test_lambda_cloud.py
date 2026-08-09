from __future__ import annotations

import importlib.util
import json
import os
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
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
    assert "ServerAliveInterval=15" in argv
    assert "ServerAliveCountMax=20" in argv


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
