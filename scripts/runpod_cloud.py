#!/usr/bin/env python3
"""SSH and file transfer for a Pod provisioned through the Runpod MCP.

Save a sanitized get-pod response to .runpod/pod.json first. Refresh it after
every restart: the external SSH port can change. No lifecycle API credential
is sent to the Pod, and this helper does not create or terminate resources.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import shlex
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/workspace/gleipnir"
EXCLUDES = (
    ".git/",
    ".runpod/",
    ".lambda/",
    ".env",
    ".env.*",
    ".venv/",
    ".cache/",
    ".cache-runtime.env",
    ".uv-cache/",
    "__pycache__/",
    ".pytest_cache/",
    ".ruff_cache/",
    ".codex/",
    ".agents/",
    ".aws/",
    "data/",
    "results/",
    "logs/",
    "wandb/",
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.pth",
)


def relative_path(value: str) -> str:
    """Accept only paths inside the local and remote repository."""
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("Transfer paths must be nonempty repository-relative paths")
    local = (ROOT / value).resolve()
    if not local.is_relative_to(ROOT):
        raise ValueError("Transfer path resolves outside the repository")
    if any(
        part in {".env", ".runpod", ".lambda", ".aws", ".codex"}
        or (part.startswith(".env.") and part != ".env.example")
        for part in path.parts
    ):
        raise ValueError("Credential and client-state paths cannot be transferred")
    return path.as_posix()


def ssh_argv(pod: dict[str, Any], key: Path) -> list[str]:
    """Build a non-interactive SSH transport from live direct-SSH metadata."""
    direct = pod.get("ssh", {}).get("direct")
    if not direct:
        raise ValueError("Pod direct SSH is not ready; refresh get-pod metadata")
    host = str(ipaddress.ip_address(direct["host"]))
    port = int(direct["port"])
    if not 1 <= port <= 65535 or direct["username"] != "root":
        raise ValueError("Unexpected SSH port or username")
    return [
        "ssh",
        "-i",
        str(key.resolve()),
        "-p",
        str(port),
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"UserKnownHostsFile={ROOT / '.runpod/known_hosts'}",
        f"root@{host}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod-file", type=Path, default=ROOT / ".runpod/pod.json")
    parser.add_argument("--key", type=Path, default=ROOT / ".runpod/id_ed25519")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("sync-code")
    sub.add_parser("bootstrap")
    sub.add_parser("probe")
    execution = sub.add_parser("exec")
    execution.add_argument("command")
    for name in ("push", "pull"):
        transfer = sub.add_parser(name)
        transfer.add_argument("path")
    args = parser.parse_args()
    pod = json.loads(args.pod_file.read_text())
    ssh = ssh_argv(pod, args.key)
    host = ssh[-1]
    transport = shlex.join(ssh[:-1])
    if args.action == "sync-code":
        subprocess.run([*ssh, f"mkdir -p {REMOTE_ROOT}"], check=True)
        subprocess.run(
            [
                "rsync",
                "-az",
                "--no-owner",
                "--no-group",
                "--partial",
                "--timeout=60",
                "-e",
                transport,
                "--include=.env.example",
                *[f"--exclude={item}" for item in EXCLUDES],
                f"{ROOT}/",
                f"{host}:{REMOTE_ROOT}/",
            ],
            check=True,
        )
    elif args.action in {"push", "pull"}:
        path = relative_path(args.path)
        remote = f"{host}:{REMOTE_ROOT}/{path}"
        local = ROOT / path
        directory = local.is_dir() if args.action == "push" else args.path.endswith("/")
        if args.action == "push":
            parent = PurePosixPath(path).parent
            subprocess.run(
                [*ssh, f"mkdir -p {shlex.quote(f'{REMOTE_ROOT}/{parent}')}"],
                check=True,
            )
        else:
            local.parent.mkdir(parents=True, exist_ok=True)
            if directory:
                local.mkdir(parents=True, exist_ok=True)
        suffix = "/" if directory else ""
        source, destination = (
            (f"{local}{suffix}", f"{remote}{suffix}")
            if args.action == "push"
            else (f"{remote}{suffix}", f"{local}{suffix}")
        )
        subprocess.run(
            [
                "rsync",
                "-az",
                "--no-owner",
                "--no-group",
                "--partial",
                "--timeout=60",
                "-e",
                transport,
                source,
                destination,
            ],
            check=True,
        )
    else:
        command = {
            "bootstrap": f"cd {REMOTE_ROOT} && bash scripts/bootstrap_runpod.sh",
            "probe": "nvidia-smi; df -h /workspace; python3 --version",
            "exec": getattr(args, "command", ""),
        }[args.action]
        subprocess.run([*ssh, command], check=True)


if __name__ == "__main__":
    main()
