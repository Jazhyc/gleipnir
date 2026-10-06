"""Retire only the identity-verified campaign server before changing kernels."""

import argparse
import json
import os
import signal
import time
from pathlib import Path

from experiments.b200_attention_gdn_serving.run import OUTPUT, ROOT, write


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-name", required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    if Path(args.archive_name).name != args.archive_name:
        raise ValueError("archive name must be a filename stem")
    metadata = OUTPUT / "server.json"
    receipt = json.loads(metadata.read_text())
    pid = receipt["pid"]
    actual = [
        value.decode()
        for value in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        if value
    ]
    if actual != receipt["command"] or os.getpgid(pid) != pid:
        raise ValueError("refuse to stop a server with a changed process identity")
    worker = json.loads((OUTPUT / "loaded_precision.json").read_text())["worker_pid"]
    os.killpg(pid, signal.SIGTERM)
    alive = []
    for _ in range(45):
        alive = []
        for value in (pid, worker):
            path = Path(f"/proc/{value}/stat")
            if path.exists() and path.read_text().split(") ")[1][0] != "Z":
                alive.append(value)
        if not alive:
            break
        time.sleep(1)
    if alive:
        raise RuntimeError(f"serving processes did not exit: {alive}")
    receipt.update(
        status="retired",
        retired_at_unix=time.time(),
        reason=args.reason,
        verified_worker_pid=worker,
    )
    write(OUTPUT / f"{args.archive_name}_server_retired.json", receipt)
    metadata.unlink()
    log = ROOT / "logs/runpod/b200_attention_gdn_serving/server.log"
    log.rename(log.with_name(f"{args.archive_name}_server.log"))
    print(f"server_retired api={pid} engine={worker}", flush=True)


if __name__ == "__main__":
    main()
