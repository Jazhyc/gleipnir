#!/usr/bin/env python3
"""Manage persistent Lambda Cloud GPU instances for experiment campaigns.

The Lambda API key is used only from the local machine for lifecycle operations.
Experiment execution and file transfer use SSH, so the API key never needs to be
copied to a cloud instance.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shlex
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn

ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / ".lambda"
KNOWN_HOSTS = STATE_DIR / "known_hosts"
SSH_CONTROL_PATH = STATE_DIR / "ssh-%C"
TCP_MSS_PROXY = ROOT / "scripts/tcp_mss_proxy.py"
DEFAULT_API_URL = "https://cloud.lambda.ai/api/v1"
INSTANCE_NAME_PREFIX = "gleipnir-"
CAMPAIGN_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,47}$")
SSH_KEY_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
REMOTE_ROOT = "gleipnir"
REMOTE_SECRETS_FILE = ".config/gleipnir/secrets.env"
REMOTE_RUNTIME_FILE = ".config/gleipnir/runtime.env"
VLLM_SMOKE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SSH_CONNECT_TIMEOUT_SECONDS = 10
SSH_CONTROL_PERSIST_SECONDS = 600
SSH_KEEPALIVE_INTERVAL_SECONDS = 15
SSH_KEEPALIVE_FAILURES = 3
SSH_TCP_MSS = 1400
RSYNC_IO_TIMEOUT_SECONDS = 60
TRANSFER_ATTEMPTS = 3
MAX_STATUS_BYTES = 1024 * 1024
ALLOWED_REMOTE_SECRETS = frozenset(
    {
        "HF_TOKEN",
        "OPENROUTER_API_KEY",
        "WANDB_API_KEY",
        "WIKIMEDIA_ACCESS_TOKEN",
    }
)
CODE_SYNC_EXCLUDES = (
    ".git/",
    ".lambda/",
    ".env",
    ".venv/",
    ".uv-cache/",
    "__pycache__/",
    "*.pyc",
    "data/",
    "logs/",
    "results/",
    "wandb/",
)


class LambdaCloudError(RuntimeError):
    """A safe-to-display Lambda Cloud or local orchestration error."""


def fail(message: str) -> NoReturn:
    raise LambdaCloudError(message)


def load_local_env(path: Path = ROOT / ".env") -> None:
    """Load simple KEY=VALUE entries without overriding the process environment."""
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def api_key_from_env() -> str:
    key = os.environ.get("LAMBDA_API_KEY", "").strip()
    if not key:
        fail("LAMBDA_API_KEY is not set (it may be placed in the git-ignored .env)")
    return key


class LambdaCloudClient:
    """Small standard-library client for the Lambda Cloud REST API."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_API_URL,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        body = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            f"{self.base_url}/{path.lstrip('/')}",
            data=body,
            method=method,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "gleipnir-lambda-runner/1",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            detail = safe_api_error(error)
            raise LambdaCloudError(
                f"Lambda API {method} {path} failed with HTTP {error.code}: {detail}"
            ) from None
        except urllib.error.URLError as error:
            raise LambdaCloudError(
                f"Lambda API {method} {path} could not be reached: {error.reason}"
            ) from None
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise LambdaCloudError(
                f"Lambda API {method} {path} returned invalid JSON: {error}"
            ) from None
        if not isinstance(result, dict) or "data" not in result:
            raise LambdaCloudError(
                f"Lambda API {method} {path} returned an unexpected response"
            )
        return result["data"]

    def instance_types(self) -> dict[str, dict[str, Any]]:
        return self.request("GET", "instance-types")

    def images(self) -> list[dict[str, Any]]:
        return self.request("GET", "images")

    def instances(self) -> list[dict[str, Any]]:
        return self.request("GET", "instances")

    def ssh_keys(self) -> list[dict[str, Any]]:
        return self.request("GET", "ssh-keys")

    def add_ssh_key(self, name: str, public_key: str) -> dict[str, Any]:
        return self.request(
            "POST",
            "ssh-keys",
            {"name": name, "public_key": public_key},
        )

    def launch(self, payload: dict[str, Any]) -> list[str]:
        response = self.request("POST", "instance-operations/launch", payload)
        return response["instance_ids"]

    def terminate(self, instance_ids: Sequence[str]) -> list[dict[str, Any]]:
        response = self.request(
            "POST",
            "instance-operations/terminate",
            {"instance_ids": list(instance_ids)},
        )
        return response["terminated_instances"]


def safe_api_error(error: urllib.error.HTTPError) -> str:
    try:
        payload = json.loads(error.read().decode())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return error.reason or "request rejected"
    error_value = payload.get("error", payload)
    if isinstance(error_value, dict):
        return str(
            error_value.get("message") or error_value.get("code") or "request rejected"
        )
    return str(error_value)[:500]


def campaign_instance_name(campaign: str) -> str:
    if not CAMPAIGN_RE.fullmatch(campaign):
        fail(
            "campaign must start with a lowercase letter or digit and contain only "
            "lowercase letters, digits, and hyphens (maximum 48 characters)"
        )
    return f"{INSTANCE_NAME_PREFIX}{campaign}"


def matching_instances(
    instances: Sequence[dict[str, Any]], campaign: str
) -> list[dict[str, Any]]:
    expected_name = campaign_instance_name(campaign)
    prefixed = [item for item in instances if item.get("name") == expected_name]
    if prefixed:
        return prefixed
    # Instances created or renamed in the Lambda console may use a human title
    # instead of this helper's ``gleipnir-`` prefix. Accept only a unique exact
    # case-insensitive title match; never use substring or fuzzy matching.
    return [
        item
        for item in instances
        if str(item.get("name", "")).casefold() == campaign.casefold()
    ]


def require_campaign_instance(
    client: LambdaCloudClient,
    campaign: str,
    *,
    require_active: bool = False,
) -> dict[str, Any]:
    matches = matching_instances(client.instances(), campaign)
    if not matches:
        fail(f"no running instance found for campaign {campaign!r}")
    if len(matches) != 1:
        fail(
            f"found {len(matches)} instances for campaign {campaign!r}; "
            "resolve the duplicate instances in Lambda before continuing"
        )
    instance = matches[0]
    if require_active and (
        instance.get("status") != "active" or not instance.get("ip")
    ):
        fail(f"campaign {campaign!r} is {instance.get('status')!r}, not ready for SSH")
    return instance


def public_key_fingerprint(public_key: str) -> str:
    fields = public_key.strip().split()
    if len(fields) < 2:
        fail("the SSH public key is malformed")
    try:
        decoded = base64.b64decode(fields[1], validate=True)
    except (ValueError, base64.binascii.Error):
        fail("the SSH public key body is not valid base64")
    digest = base64.b64encode(hashlib.sha256(decoded).digest()).decode().rstrip("=")
    return f"SHA256:{digest}"


def default_private_key_path() -> Path:
    configured = os.environ.get("LAMBDA_SSH_KEY_PATH")
    return (
        Path(configured).expanduser() if configured else Path.home() / ".ssh/id_ed25519"
    )


def public_key_for_private_key(private_key: Path) -> str:
    public_path = Path(f"{private_key}.pub")
    if public_path.exists():
        return public_path.read_text().strip()
    result = subprocess.run(
        ["ssh-keygen", "-y", "-f", str(private_key)],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or "ssh-keygen failed"
        fail(f"could not derive a public key from {private_key}: {detail}")
    return result.stdout.strip()


def validate_private_key(private_key: Path) -> None:
    if not private_key.is_file():
        fail(f"SSH private key not found: {private_key}")
    mode = stat.S_IMODE(private_key.stat().st_mode)
    if mode & 0o077:
        fail(f"SSH private key permissions are too broad ({mode:o}); run chmod 600")


def registered_key_name(
    client: LambdaCloudClient,
    private_key: Path,
) -> str:
    validate_private_key(private_key)
    fingerprint = public_key_fingerprint(public_key_for_private_key(private_key))
    matches = [
        key["name"]
        for key in client.ssh_keys()
        if public_key_fingerprint(key["public_key"]) == fingerprint
    ]
    configured = os.environ.get("LAMBDA_SSH_KEY_NAME")
    if configured:
        if configured not in matches:
            fail(
                f"LAMBDA_SSH_KEY_NAME={configured!r} does not match "
                f"the local key fingerprint {fingerprint}"
            )
        return configured
    if not matches:
        fail(
            f"local SSH key {private_key} ({fingerprint}) is not registered; "
            "run register-key --name <name>"
        )
    if len(matches) > 1:
        fail(
            "the local SSH key is registered under multiple names; set "
            "LAMBDA_SSH_KEY_NAME explicitly"
        )
    return matches[0]


def display_instance_type(name: str, item: dict[str, Any]) -> dict[str, Any]:
    details = item["instance_type"]
    specs = details["specs"]
    return {
        "name": name,
        "gpu": details["gpu_description"],
        "architecture": details.get("architecture"),
        "gpus": specs["gpus"],
        "vcpus": specs["vcpus"],
        "memory_gib": specs["memory_gib"],
        "storage_gib": specs["storage_gib"],
        "usd_per_hour": details["price_cents_per_hour"] / 100,
        "regions": [
            region["name"] for region in item["regions_with_capacity_available"]
        ],
    }


def display_instance(instance: dict[str, Any]) -> dict[str, Any]:
    instance_type = instance.get("instance_type", {})
    return {
        "id": instance["id"],
        "name": instance.get("name"),
        "status": instance["status"],
        "ip": instance.get("ip"),
        "hostname": instance.get("hostname"),
        "instance_type": instance_type.get("name"),
        "gpu": instance_type.get("gpu_description"),
        "usd_per_hour": (
            instance_type["price_cents_per_hour"] / 100
            if "price_cents_per_hour" in instance_type
            else None
        ),
        "region": instance.get("region", {}).get("name"),
        "ssh_key_names": instance.get("ssh_key_names", []),
        "file_system_names": instance.get("file_system_names", []),
    }


def print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=False))


def command_types(args: argparse.Namespace, client: LambdaCloudClient) -> None:
    rows = []
    for name, item in client.instance_types().items():
        row = display_instance_type(name, item)
        if args.available_only and not row["regions"]:
            continue
        if (
            args.min_gpu_memory
            and extract_gpu_memory_gib(row["gpu"]) < args.min_gpu_memory
        ):
            continue
        rows.append(row)
    rows.sort(
        key=lambda row: (not bool(row["regions"]), row["usd_per_hour"], row["name"])
    )
    print_json(rows)


def extract_gpu_memory_gib(description: str) -> int:
    match = re.search(r"\((\d+)\s*GB\b", description)
    return int(match.group(1)) if match else 0


def command_instances(args: argparse.Namespace, client: LambdaCloudClient) -> None:
    instances = client.instances()
    if args.campaign:
        instances = matching_instances(instances, args.campaign)
    print_json([display_instance(item) for item in instances])


def command_images(args: argparse.Namespace, client: LambdaCloudClient) -> None:
    images = client.images()
    if args.region:
        images = [
            image
            for image in images
            if image.get("region", {}).get("name") == args.region
        ]
    if args.family:
        images = [image for image in images if image.get("family") == args.family]
    print_json(
        [
            {
                "id": image["id"],
                "name": image["name"],
                "family": image["family"],
                "version": image["version"],
                "architecture": image["architecture"],
                "region": image["region"]["name"],
                "updated_time": image["updated_time"],
            }
            for image in images
        ]
    )


def command_ssh_keys(_args: argparse.Namespace, client: LambdaCloudClient) -> None:
    print_json(
        [
            {
                "name": key["name"],
                "id": key["id"],
                "fingerprint": public_key_fingerprint(key["public_key"]),
            }
            for key in client.ssh_keys()
        ]
    )


def command_register_key(args: argparse.Namespace, client: LambdaCloudClient) -> None:
    if not SSH_KEY_NAME_RE.fullmatch(args.name):
        fail("SSH key name contains unsupported characters or is too long")
    private_key = Path(args.private_key).expanduser()
    validate_private_key(private_key)
    public_key = public_key_for_private_key(private_key)
    fingerprint = public_key_fingerprint(public_key)
    keys = client.ssh_keys()
    same_name = [key for key in keys if key["name"] == args.name]
    if same_name:
        existing_fingerprint = public_key_fingerprint(same_name[0]["public_key"])
        if existing_fingerprint == fingerprint:
            print_json(
                {
                    "status": "already_registered",
                    "name": args.name,
                    "fingerprint": fingerprint,
                }
            )
            return
        fail(f"Lambda already has a different SSH key named {args.name!r}")
    same_key = [
        key["name"]
        for key in keys
        if public_key_fingerprint(key["public_key"]) == fingerprint
    ]
    if same_key:
        print_json(
            {
                "status": "already_registered",
                "name": same_key[0],
                "fingerprint": fingerprint,
            }
        )
        return
    created = client.add_ssh_key(args.name, public_key)
    print_json(
        {
            "status": "registered",
            "name": created["name"],
            "id": created["id"],
            "fingerprint": fingerprint,
        }
    )


def command_doctor(args: argparse.Namespace, client: LambdaCloudClient) -> None:
    private_key = Path(args.private_key).expanduser()
    api_types = client.instance_types()
    key_name = registered_key_name(client, private_key)
    available = sum(
        bool(item["regions_with_capacity_available"]) for item in api_types.values()
    )
    print_json(
        {
            "api": "ok",
            "available_instance_types": available,
            "private_key": str(private_key),
            "registered_ssh_key_name": key_name,
            "running_instances": len(client.instances()),
        }
    )


def command_launch(args: argparse.Namespace, client: LambdaCloudClient) -> None:
    campaign_name = campaign_instance_name(args.campaign)
    existing = matching_instances(client.instances(), args.campaign)
    if existing:
        if len(existing) > 1:
            fail(f"multiple instances already use campaign name {campaign_name!r}")
        print_json(
            {
                "status": "already_exists",
                "instance": display_instance(existing[0]),
            }
        )
        return
    if not args.yes:
        fail(
            "launch creates a billable GPU instance; pass --yes after "
            "reviewing the price"
        )
    types = client.instance_types()
    if args.instance_type not in types:
        fail(f"unknown Lambda instance type: {args.instance_type}")
    item = types[args.instance_type]
    row = display_instance_type(args.instance_type, item)
    if args.region not in row["regions"]:
        fail(
            f"{args.instance_type} currently has no reported capacity in "
            f"{args.region}; "
            f"available regions: {row['regions'] or 'none'}"
        )
    if row["architecture"] != "x86_64" and not args.allow_non_x86:
        fail(
            f"{args.instance_type} uses architecture {row['architecture']!r}; "
            "pass --allow-non-x86 only after checking wheel compatibility"
        )
    private_key = Path(args.private_key).expanduser()
    local_key_name = registered_key_name(client, private_key)
    if args.ssh_key_name and args.ssh_key_name != local_key_name:
        fail(
            f"--ssh-key-name {args.ssh_key_name!r} does not match this system's "
            f"registered key {local_key_name!r}"
        )
    ssh_key_name = args.ssh_key_name or local_key_name
    payload: dict[str, Any] = {
        "region_name": args.region,
        "instance_type_name": args.instance_type,
        "ssh_key_names": [ssh_key_name],
        "file_system_names": args.file_system,
        "quantity": 1,
        "name": campaign_name,
        "tags": [
            {"key": "project", "value": "gleipnir"},
            {"key": "campaign", "value": args.campaign},
            {"key": "managed-by", "value": "lambda-cloud-script"},
        ],
    }
    if args.image_family:
        payload["image"] = {"family": args.image_family}
    instance_ids = client.launch(payload)
    print_json(
        {
            "status": "launch_requested",
            "campaign": args.campaign,
            "instance_ids": instance_ids,
            "instance_type": row,
            "ssh_key_name": ssh_key_name,
            "image_family": args.image_family or "Lambda default",
            "retention": "explicit termination only",
        }
    )


def command_wait(args: argparse.Namespace, client: LambdaCloudClient) -> None:
    deadline = time.monotonic() + args.timeout
    last_status = None
    while True:
        instance = require_campaign_instance(client, args.campaign)
        status = instance["status"]
        if status != last_status:
            print(f"{args.campaign}: {status}", file=sys.stderr, flush=True)
            last_status = status
        if status == "active" and instance.get("ip"):
            print_json(display_instance(instance))
            return
        if status in {"unhealthy", "terminated", "terminating", "preempted"}:
            fail(f"campaign instance entered terminal status {status!r}")
        if time.monotonic() >= deadline:
            fail(f"timed out after {args.timeout}s waiting for campaign instance")
        time.sleep(args.interval)


def ssh_transport_argv(private_key: Path) -> list[str]:
    """Return the shared, fail-fast SSH transport configuration."""
    STATE_DIR.mkdir(exist_ok=True)
    proxy_command = shlex.join(
        [
            sys.executable,
            str(TCP_MSS_PROXY),
            "%h",
            "%p",
            "--mss",
            str(SSH_TCP_MSS),
        ]
    )
    return [
        "-i",
        str(private_key),
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"UserKnownHostsFile={KNOWN_HOSTS}",
        "-o",
        f"ConnectTimeout={SSH_CONNECT_TIMEOUT_SECONDS}",
        "-o",
        "ConnectionAttempts=3",
        "-o",
        f"ServerAliveInterval={SSH_KEEPALIVE_INTERVAL_SECONDS}",
        "-o",
        f"ServerAliveCountMax={SSH_KEEPALIVE_FAILURES}",
        "-o",
        "TCPKeepAlive=yes",
        "-o",
        f"ProxyCommand={proxy_command}",
        "-o",
        "ControlMaster=auto",
        "-o",
        f"ControlPersist={SSH_CONTROL_PERSIST_SECONDS}",
        "-o",
        f"ControlPath={SSH_CONTROL_PATH}",
    ]


def ssh_argv(private_key: Path, ip: str) -> list[str]:
    return [
        "ssh",
        *ssh_transport_argv(private_key),
        f"ubuntu@{ip}",
    ]


def active_ssh_target(
    args: argparse.Namespace,
    client: LambdaCloudClient,
) -> tuple[Path, dict[str, Any]]:
    private_key = Path(args.private_key).expanduser()
    validate_private_key(private_key)
    instance = require_campaign_instance(client, args.campaign, require_active=True)
    return private_key, instance


def run_checked(argv: Sequence[str], *, input_text: str | None = None) -> None:
    result = subprocess.run(
        list(argv),
        text=True,
        input=input_text,
        check=False,
    )
    if result.returncode != 0:
        fail(f"command failed with exit code {result.returncode}: {shlex.join(argv)}")


def run_transfer(argv: Sequence[str], *, attempts: int = TRANSFER_ATTEMPTS) -> None:
    """Run resumable transfer commands with bounded exponential-backoff retries."""
    if attempts < 1:
        raise ValueError("transfer attempts must be positive")
    last_returncode = 0
    for attempt in range(1, attempts + 1):
        result = subprocess.run(list(argv), check=False)
        last_returncode = result.returncode
        if result.returncode == 0:
            return
        if attempt < attempts:
            delay = 2 ** (attempt - 1)
            print(
                f"transfer failed with exit code {result.returncode} "
                f"(attempt {attempt}/{attempts}); retrying in {delay}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
    fail(
        f"transfer failed after {attempts} attempts with exit code {last_returncode}: "
        f"{shlex.join(argv)}"
    )


def command_ssh(args: argparse.Namespace, client: LambdaCloudClient) -> None:
    private_key, instance = active_ssh_target(args, client)
    argv = ssh_argv(private_key, instance["ip"])
    if args.command:
        command = args.command
        if command[0] == "--":
            command = command[1:]
        if command:
            argv.append(shlex.join(command))
    run_checked(argv)


def command_status(args: argparse.Namespace, client: LambdaCloudClient) -> None:
    """Read one small atomic JSON status artifact without streaming active logs."""
    private_key, instance = active_ssh_target(args, client)
    remote_path = PurePosixPath(REMOTE_ROOT) / ensure_relative_remote_path(
        args.remote_path
    )
    remote_program = f"""import json
import sys
from pathlib import Path

path = Path.home() / sys.argv[1]
size = path.stat().st_size
if size > {MAX_STATUS_BYTES}:
    raise SystemExit(f"status file is too large: {{size}} bytes")
with path.open(encoding="utf-8") as handle:
    value = json.load(handle)
print(json.dumps(value, indent=2, sort_keys=True))
"""
    command = shlex.join(["python3", "-c", remote_program, str(remote_path)])
    argv = ssh_argv(private_key, instance["ip"]) + [command]
    run_checked(argv)


def command_probe(args: argparse.Namespace, client: LambdaCloudClient) -> None:
    private_key, instance = active_ssh_target(args, client)
    remote_script = """set -eu
printf 'host_arch='
uname -m
printf 'os='
. /etc/os-release
printf '%s %s\\n' "$NAME" "$VERSION_ID"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
printf 'root_free='
df -h / | awk 'NR == 2 {print $4}'
"""
    argv = ssh_argv(private_key, instance["ip"]) + ["bash", "-s"]
    run_checked(argv, input_text=remote_script)


def command_compute_probe(args: argparse.Namespace, client: LambdaCloudClient) -> None:
    private_key, instance = active_ssh_target(args, client)
    remote_script = f"""set -eu
source "$HOME/{REMOTE_RUNTIME_FILE}"
cd "$HOME/{REMOTE_ROOT}"
python - <<'PY'
import time

import torch

size = 8192
dtype = torch.float16
torch.manual_seed(0)
left = torch.randn((size, size), device="cuda", dtype=dtype)
right = torch.randn((size, size), device="cuda", dtype=dtype)
for _ in range(2):
    torch.mm(left, right)
torch.cuda.synchronize()
started = time.perf_counter()
result = torch.mm(left, right)
torch.cuda.synchronize()
elapsed = time.perf_counter() - started
tflops = (2 * size**3) / elapsed / 1e12
print(f"torch={{torch.__version__}}")
print(f"gpu={{torch.cuda.get_device_name(0)}}")
print(f"matrix_size={{size}}")
print(f"elapsed_seconds={{elapsed:.6f}}")
print(f"estimated_tflops={{tflops:.1f}}")
print(f"checksum={{result[0, 0].item():.6f}}")
print(f"peak_allocated_gib={{torch.cuda.max_memory_allocated() / 2**30:.2f}}")
PY
"""
    argv = ssh_argv(private_key, instance["ip"]) + ["bash", "-s"]
    run_checked(argv, input_text=remote_script)


def command_vllm_smoke(args: argparse.Namespace, client: LambdaCloudClient) -> None:
    private_key, instance = active_ssh_target(args, client)
    remote_script = f"""set -euo pipefail
source "$HOME/{REMOTE_RUNTIME_FILE}"
if [ -f "$HOME/{REMOTE_SECRETS_FILE}" ]; then
    source "$HOME/{REMOTE_SECRETS_FILE}"
fi
export PYTHONUNBUFFERED=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
cd "$HOME/{REMOTE_ROOT}"
mkdir -p logs/lambda/vllm-smoke
log_path="logs/lambda/vllm-smoke/latest.log"
status_path="logs/lambda/vllm-smoke/latest.status"
set +e
.venv/bin/python - >"$log_path" 2>&1 <<'PY'
from vllm import LLM, SamplingParams

model_id = "{VLLM_SMOKE_MODEL}"
model = LLM(
    model=model_id,
    dtype="bfloat16",
    max_model_len=512,
    gpu_memory_utilization=0.20,
    enforce_eager=True,
)
outputs = model.generate(
    ["Reply with exactly LAMBDA_OK."],
    SamplingParams(temperature=0.0, max_tokens=16),
)
text = outputs[0].outputs[0].text.strip()
if not text:
    raise SystemExit("vLLM returned an empty completion")
print(f"model={{model_id}}")
print(f"completion={{text!r}}")
print("vllm_smoke=ok")
PY
status=$?
set -e
printf '%s\\n' "$status" >"$status_path"
if [ "$status" -ne 0 ]; then
    tail -n 40 "$log_path" >&2
    printf 'vllm_smoke_exit=%s\\n' "$status" >&2
    exit "$status"
fi
cat "$log_path"
"""
    argv = ssh_argv(private_key, instance["ip"]) + ["bash", "-s"]
    run_checked(argv, input_text=remote_script)


def ensure_repo_relative_local_path(path_text: str) -> Path:
    path = (ROOT / path_text).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError:
        fail("local sync paths must remain inside the repository")
    return path


def ensure_relative_remote_path(path_text: str) -> PurePosixPath:
    path = PurePosixPath(path_text)
    if path.is_absolute() or ".." in path.parts:
        fail("remote sync paths must be relative and cannot contain '..'")
    if not path.parts:
        fail("remote sync path cannot be empty")
    return path


def rsync_ssh_command(private_key: Path) -> str:
    return shlex.join(["ssh", *ssh_transport_argv(private_key)])


def rsync_argv(private_key: Path) -> list[str]:
    """Return shared rsync settings for resumable SSH transfers."""
    return [
        "rsync",
        "-a",
        "--partial",
        "--partial-dir=.rsync-partial",
        f"--timeout={RSYNC_IO_TIMEOUT_SECONDS}",
        "--info=progress2",
        "-e",
        rsync_ssh_command(private_key),
    ]


def command_sync_code(args: argparse.Namespace, client: LambdaCloudClient) -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if status.returncode != 0:
        fail(f"could not inspect the Git working tree: {status.stderr.strip()}")
    if status.stdout.strip() and not args.include_uncommitted:
        fail(
            "the working tree has uncommitted content; use sync-commit for a "
            "reproducible HEAD snapshot, or pass --include-uncommitted explicitly"
        )
    private_key, instance = active_ssh_target(args, client)
    target = f"ubuntu@{instance['ip']}:{REMOTE_ROOT}/"
    mkdir_argv = ssh_argv(private_key, instance["ip"]) + ["mkdir", "-p", REMOTE_ROOT]
    run_checked(mkdir_argv)
    argv = rsync_argv(private_key)
    for pattern in CODE_SYNC_EXCLUDES:
        argv.extend(["--exclude", pattern])
    argv.extend([f"{ROOT}/", target])
    run_transfer(argv)


def command_sync_commit(args: argparse.Namespace, client: LambdaCloudClient) -> None:
    private_key, instance = active_ssh_target(args, client)
    mkdir_argv = ssh_argv(private_key, instance["ip"]) + [
        "mkdir",
        "-p",
        REMOTE_ROOT,
    ]
    run_checked(mkdir_argv)
    archive = subprocess.Popen(
        ["git", "archive", "--format=tar", args.revision],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert archive.stdout is not None
    remote_argv = ssh_argv(private_key, instance["ip"]) + [
        "tar",
        "-xf",
        "-",
        "-C",
        REMOTE_ROOT,
    ]
    remote = subprocess.run(remote_argv, stdin=archive.stdout, check=False)
    archive.stdout.close()
    archive_stderr = archive.stderr.read().decode() if archive.stderr else ""
    archive_code = archive.wait()
    if archive_code != 0:
        fail(
            f"git archive failed for revision {args.revision!r}: "
            f"{archive_stderr.strip() or f'exit code {archive_code}'}"
        )
    if remote.returncode != 0:
        fail(f"remote archive extraction failed with exit code {remote.returncode}")
    revision = subprocess.run(
        ["git", "rev-parse", args.revision],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if revision.returncode != 0:
        fail(f"could not resolve revision {args.revision!r}")
    print_json(
        {
            "status": "synced",
            "campaign": args.campaign,
            "revision": revision.stdout.strip(),
            "remote_root": f"~/{REMOTE_ROOT}",
            "uncommitted_content": "excluded",
        }
    )


def command_bootstrap(args: argparse.Namespace, client: LambdaCloudClient) -> None:
    private_key, instance = active_ssh_target(args, client)
    cuda_compat_export = (
        'export LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat'
        '${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"'
    )
    remote_script = f"""set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="$HOME/.cache/uv"
if [ -d /usr/local/cuda-13.0/compat ]; then
    {cuda_compat_export}
fi
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/{REMOTE_ROOT}"
uv python install 3.12
UV_PROJECT_ENVIRONMENT=.venv uv sync --python 3.12 --locked
umask 077
mkdir -p "$HOME/.config/gleipnir" "$HOME/.cache/gleipnir"
cat > "$HOME/{REMOTE_RUNTIME_FILE}" <<'EOF'
export PATH="$HOME/{REMOTE_ROOT}/.venv/bin:$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="$HOME/.cache/uv"
export SCRATCH="$HOME/.cache/gleipnir"
export HF_HOME="$SCRATCH/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
if [ -d /usr/local/cuda-13.0/compat ]; then
    {cuda_compat_export}
fi
EOF
chmod 600 "$HOME/{REMOTE_RUNTIME_FILE}"
.venv/bin/python - <<'PY'
import platform
import torch
import vllm

print(f"python={{platform.python_version()}}")
print(f"torch={{torch.__version__}}")
print(f"vllm={{vllm.__version__}}")
print(f"cuda_available={{torch.cuda.is_available()}}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to PyTorch")
print(f"gpu={{torch.cuda.get_device_name(0)}}")
print(f"cuda_runtime={{torch.version.cuda}}")
PY
"""
    argv = ssh_argv(private_key, instance["ip"]) + ["bash", "-s"]
    run_checked(argv, input_text=remote_script)


def command_push(args: argparse.Namespace, client: LambdaCloudClient) -> None:
    private_key, instance = active_ssh_target(args, client)
    local_path = ensure_repo_relative_local_path(args.local_path)
    if not local_path.exists():
        fail(f"local path does not exist: {local_path}")
    remote_path = ensure_relative_remote_path(args.remote_path or args.local_path)
    remote_parent = PurePosixPath(REMOTE_ROOT) / remote_path.parent
    mkdir_argv = ssh_argv(private_key, instance["ip"]) + [
        "mkdir",
        "-p",
        str(remote_parent),
    ]
    run_checked(mkdir_argv)
    source = f"{local_path}/" if local_path.is_dir() else str(local_path)
    target_path = PurePosixPath(REMOTE_ROOT) / remote_path
    target = f"ubuntu@{instance['ip']}:{target_path}"
    if local_path.is_dir():
        target += "/"
    argv = [*rsync_argv(private_key), source, target]
    run_transfer(argv)


def command_pull(args: argparse.Namespace, client: LambdaCloudClient) -> None:
    private_key, instance = active_ssh_target(args, client)
    remote_path = ensure_relative_remote_path(args.remote_path)
    local_path = ensure_repo_relative_local_path(args.local_path or args.remote_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    source_path = PurePosixPath(REMOTE_ROOT) / remote_path
    source = f"ubuntu@{instance['ip']}:{source_path}"
    if args.directory:
        source += "/"
        local_path.mkdir(parents=True, exist_ok=True)
        target = f"{local_path}/"
    else:
        target = str(local_path)
    argv = [*rsync_argv(private_key), source, target]
    run_transfer(argv)


def build_remote_secret_payload(names: Sequence[str]) -> str:
    requested = set(names)
    forbidden = sorted(requested - ALLOWED_REMOTE_SECRETS)
    if forbidden:
        fail(
            "refusing to transfer non-allowlisted secret variables: "
            + ", ".join(forbidden)
        )
    missing = sorted(name for name in requested if not os.environ.get(name))
    if missing:
        fail("required local secret variables are unset: " + ", ".join(missing))
    lines = [
        "# Generated by scripts/lambda_cloud.py; do not copy into the repository.",
    ]
    for name in sorted(requested):
        value = os.environ[name]
        if "\0" in value:
            fail(f"{name} contains an unsupported null byte")
        lines.append(f"export {name}={shlex.quote(value)}")
    return "\n".join(lines) + "\n"


def command_sync_secrets(args: argparse.Namespace, client: LambdaCloudClient) -> None:
    if not args.name:
        fail(
            "no secrets selected; pass --name with one or more of: "
            + ", ".join(sorted(ALLOWED_REMOTE_SECRETS))
        )
    private_key, instance = active_ssh_target(args, client)
    payload = build_remote_secret_payload(args.name)
    remote_script = f"""set -eu
umask 077
mkdir -p "$HOME/.config/gleipnir"
cat > "$HOME/{REMOTE_SECRETS_FILE}"
chmod 600 "$HOME/{REMOTE_SECRETS_FILE}"
"""
    argv = ssh_argv(private_key, instance["ip"]) + [
        f"bash -c {shlex.quote(remote_script)}"
    ]
    run_checked(argv, input_text=payload)
    print_json(
        {
            "status": "synced",
            "campaign": args.campaign,
            "variables": sorted(set(args.name)),
            "remote_file": f"~/{REMOTE_SECRETS_FILE}",
            "excluded": ["LAMBDA_API_KEY"],
        }
    )


def command_terminate(args: argparse.Namespace, client: LambdaCloudClient) -> None:
    instance = require_campaign_instance(client, args.campaign)
    if not args.yes:
        fail(
            "termination permanently erases instance-local data; pull artifacts, "
            "then pass --yes"
        )
    terminated = client.terminate([instance["id"]])
    print_json(
        {
            "status": "termination_requested",
            "campaign": args.campaign,
            "instances": [display_instance(item) for item in terminated],
        }
    )


def add_private_key_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--private-key",
        default=str(default_private_key_path()),
        help=(
            "local SSH private key (default: LAMBDA_SSH_KEY_PATH or ~/.ssh/id_ed25519)"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage persistent Lambda Cloud experiment campaign instances.",
    )
    parser.add_argument(
        "--api-url",
        default=os.environ.get("LAMBDA_API_URL", DEFAULT_API_URL),
        help=argparse.SUPPRESS,
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    types_parser = subparsers.add_parser(
        "types", help="list GPU types and live capacity"
    )
    types_parser.add_argument("--available-only", action="store_true")
    types_parser.add_argument("--min-gpu-memory", type=int, default=0)
    types_parser.set_defaults(handler=command_types)

    instances_parser = subparsers.add_parser("instances", help="list running instances")
    instances_parser.add_argument("--campaign")
    instances_parser.set_defaults(handler=command_instances)

    images_parser = subparsers.add_parser(
        "images", help="list available Lambda base images"
    )
    images_parser.add_argument("--region")
    images_parser.add_argument("--family")
    images_parser.set_defaults(handler=command_images)

    keys_parser = subparsers.add_parser("ssh-keys", help="list safe SSH key metadata")
    keys_parser.set_defaults(handler=command_ssh_keys)

    register_parser = subparsers.add_parser(
        "register-key", help="idempotently register this system's public SSH key"
    )
    register_parser.add_argument("--name", required=True)
    add_private_key_argument(register_parser)
    register_parser.set_defaults(handler=command_register_key)

    doctor_parser = subparsers.add_parser(
        "doctor", help="check API authentication and local SSH-key registration"
    )
    add_private_key_argument(doctor_parser)
    doctor_parser.set_defaults(handler=command_doctor)

    launch_parser = subparsers.add_parser(
        "launch", help="launch one persistent, named campaign instance"
    )
    launch_parser.add_argument("--campaign", required=True)
    launch_parser.add_argument("--instance-type", required=True)
    launch_parser.add_argument("--region", required=True)
    launch_parser.add_argument("--ssh-key-name")
    launch_parser.add_argument("--file-system", action="append", default=[])
    launch_parser.add_argument("--image-family")
    launch_parser.add_argument("--allow-non-x86", action="store_true")
    launch_parser.add_argument(
        "--yes",
        action="store_true",
        help="confirm launch of the displayed billable resource",
    )
    add_private_key_argument(launch_parser)
    launch_parser.set_defaults(handler=command_launch)

    wait_parser = subparsers.add_parser(
        "wait", help="wait until a campaign is SSH-ready"
    )
    wait_parser.add_argument("--campaign", required=True)
    wait_parser.add_argument("--timeout", type=int, default=600)
    wait_parser.add_argument("--interval", type=int, default=10)
    wait_parser.set_defaults(handler=command_wait)

    ssh_parser = subparsers.add_parser("ssh", help="connect or run a command over SSH")
    ssh_parser.add_argument("--campaign", required=True)
    add_private_key_argument(ssh_parser)
    ssh_parser.add_argument("command", nargs=argparse.REMAINDER)
    ssh_parser.set_defaults(handler=command_ssh)

    status_parser = subparsers.add_parser(
        "status", help="read one bounded JSON status artifact over SSH"
    )
    status_parser.add_argument("--campaign", required=True)
    status_parser.add_argument("--remote-path", required=True)
    add_private_key_argument(status_parser)
    status_parser.set_defaults(handler=command_status)

    probe_parser = subparsers.add_parser("probe", help="check remote OS, GPU, and disk")
    probe_parser.add_argument("--campaign", required=True)
    add_private_key_argument(probe_parser)
    probe_parser.set_defaults(handler=command_probe)

    compute_parser = subparsers.add_parser(
        "compute-probe", help="run a fixed PyTorch matrix multiply on the remote GPU"
    )
    compute_parser.add_argument("--campaign", required=True)
    add_private_key_argument(compute_parser)
    compute_parser.set_defaults(handler=command_compute_probe)

    vllm_parser = subparsers.add_parser(
        "vllm-smoke",
        help=f"run fixed inference with {VLLM_SMOKE_MODEL}",
    )
    vllm_parser.add_argument("--campaign", required=True)
    add_private_key_argument(vllm_parser)
    vllm_parser.set_defaults(handler=command_vllm_smoke)

    sync_parser = subparsers.add_parser(
        "sync-code", help="copy code while excluding secrets, caches, logs, and results"
    )
    sync_parser.add_argument("--campaign", required=True)
    sync_parser.add_argument(
        "--include-uncommitted",
        action="store_true",
        help="explicitly allow transfer when the working tree is dirty",
    )
    add_private_key_argument(sync_parser)
    sync_parser.set_defaults(handler=command_sync_code)

    commit_parser = subparsers.add_parser(
        "sync-commit",
        help="copy a reproducible committed Git snapshot without working-tree changes",
    )
    commit_parser.add_argument("--campaign", required=True)
    commit_parser.add_argument("--revision", default="HEAD")
    add_private_key_argument(commit_parser)
    commit_parser.set_defaults(handler=command_sync_commit)

    bootstrap_parser = subparsers.add_parser(
        "bootstrap",
        help="install Python 3.12, sync the locked environment, and check CUDA",
    )
    bootstrap_parser.add_argument("--campaign", required=True)
    add_private_key_argument(bootstrap_parser)
    bootstrap_parser.set_defaults(handler=command_bootstrap)

    push_parser = subparsers.add_parser(
        "push", help="copy one explicit repository-relative path to the campaign"
    )
    push_parser.add_argument("--campaign", required=True)
    push_parser.add_argument("--local-path", required=True)
    push_parser.add_argument("--remote-path")
    add_private_key_argument(push_parser)
    push_parser.set_defaults(handler=command_push)

    pull_parser = subparsers.add_parser(
        "pull", help="copy one explicit campaign path back into the repository"
    )
    pull_parser.add_argument("--campaign", required=True)
    pull_parser.add_argument("--remote-path", required=True)
    pull_parser.add_argument("--local-path")
    pull_parser.add_argument(
        "--directory",
        action="store_true",
        help="treat the source and destination as directory contents",
    )
    add_private_key_argument(pull_parser)
    pull_parser.set_defaults(handler=command_pull)

    secrets_parser = subparsers.add_parser(
        "sync-secrets",
        help="transfer explicitly selected allowlisted credentials over SSH",
    )
    secrets_parser.add_argument("--campaign", required=True)
    secrets_parser.add_argument(
        "--name",
        action="append",
        choices=sorted(ALLOWED_REMOTE_SECRETS),
        default=[],
    )
    add_private_key_argument(secrets_parser)
    secrets_parser.set_defaults(handler=command_sync_secrets)

    terminate_parser = subparsers.add_parser(
        "terminate", help="explicitly terminate a campaign after collecting artifacts"
    )
    terminate_parser.add_argument("--campaign", required=True)
    terminate_parser.add_argument("--yes", action="store_true")
    terminate_parser.set_defaults(handler=command_terminate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_local_env()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        client = LambdaCloudClient(api_key_from_env(), base_url=args.api_url)
        args.handler(args, client)
    except LambdaCloudError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
