"""Retire only the recorded migration server, leaving capacity and caches intact."""

import argparse
import json
import os
import signal
import time
from pathlib import Path

from experiments.b200_inference_benchmark.run import ROOT, write


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-name", required=True)
    args = parser.parse_args()
    if Path(args.archive_name).name != args.archive_name:
        raise ValueError("archive name must be a stem")
    active = ROOT / "results/b200_attention_gdn_serving/server.json"
    receipt = json.loads(active.read_text())
    if receipt.get("runtime_migration", {}).get("vllm") != "0.31.0":
        raise ValueError("recorded server is not the migration candidate")
    pid = receipt["pid"]
    actual = [
        value.decode()
        for value in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        if value
    ]
    if actual != receipt["command"] or os.getpgid(pid) != pid:
        raise ValueError("server process identity changed")
    worker = json.loads((active.parent / "loaded_precision.json").read_text())[
        "worker_pid"
    ]
    os.killpg(pid, signal.SIGTERM)
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
        raise RuntimeError(f"candidate processes did not exit: {alive}")
    receipt.update(status="retired", retired_at_unix=time.time(), worker_pid=worker)
    write(ROOT / "results/b200_vllm031" / f"{args.archive_name}_retired.json", receipt)
    active.unlink()
    print("vllm031_retired", pid, worker, flush=True)


if __name__ == "__main__":
    main()
