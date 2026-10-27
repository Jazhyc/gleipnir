"""Serve the base Qwen3.5-4B through the recorded Gleipnir 4B v0.4 optimized command.

The release launcher's ``prepare`` builds the command and environment (runtime,
source bindings, receipts, 32K recorded context); only ``--model`` changes to
the base snapshot. ``--lens`` adds the eager Lens capture bridge, and
``--truncate`` swaps in the worker that stops the decoder after block 15 and
returns the un-normalized residual stream; ``--probe-bank`` (with truncation)
also replaces the classify pooler with the full01 probe banks.
The server runs in the foreground; the attempt log is under ``--log``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OVERLAY = "/tmp/gleipnir-vllm-lens-1.3.0"
WORKER = "experiments.b200_gdn_fast.worker.FastGdnWorker"
TRUNCATED = "experiments.probe_optimized_serving.truncate_worker.TruncatedFastGdnWorker"
WORKER_SOURCE = "experiments/probe_optimized_serving/truncate_worker.py"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, help="base model snapshot dir")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--lens", action="store_true")
    parser.add_argument("--truncate", action="store_true")
    parser.add_argument("--probe-bank", help="full01 NTP sweep dir")
    parser.add_argument("--epochs", default="2,4,8,16")
    parser.add_argument("--variants", default="fp8,raw")
    parser.add_argument("--select", help="comma-separated probe names (one epoch)")
    parser.add_argument("--log", required=True)
    args = parser.parse_args()

    from experiments.release_serving.serve import prepare
    from experiments.release_serving.targets import RECORDED_CONTEXT, TARGETS
    from gleipnir.data.monitoring import file_hash

    command, env = prepare(TARGETS["4b"], "127.0.0.1", args.port, RECORDED_CONTEXT)
    command[command.index("--model") + 1] = args.snapshot
    if args.truncate:
        worker = command.index("--worker-cls") + 1
        if command[worker] != WORKER:
            raise ValueError(f"unexpected selected worker {command[worker]}")
        command[worker] = TRUNCATED
        # Bind the worker source into the compile identity, as the release
        # launcher does for its prefix-cached workers.
        index = command.index("--additional-config") + 1
        additional = json.loads(command[index])
        additional["gleipnir_frost_fp4"][WORKER_SOURCE] = file_hash(
            ROOT / WORKER_SOURCE
        )
        command[index] = json.dumps(additional, sort_keys=True)
        env["GLEIPNIR_PROBE_RECEIPT"] = str(Path(args.log).with_suffix(".probe.json"))
        if args.probe_bank:
            env["GLEIPNIR_PROBE_BANK"] = str(ROOT / args.probe_bank)
            env["GLEIPNIR_PROBE_EPOCHS"] = args.epochs
            env["GLEIPNIR_PROBE_VARIANTS"] = args.variants
            if args.select:
                env["GLEIPNIR_PROBE_SELECT"] = args.select
    elif args.probe_bank:
        raise ValueError("the probe pooler requires --truncate")
    if args.lens:
        command[2] = "experiments.b200_vllm_lens.server"
        command += [
            "--enforce-eager",
            "--worker-extension-cls",
            "gleipnir.serving.lens_worker.MonitorLensExtension",
        ]
        env["PYTHONPATH"] += ":" + OVERLAY
        env["VLLM_LENS_DISABLE"] = "1"
    record = Path(args.log).with_suffix(".command.json")
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps({"command": command}, indent=2) + "\n")
    with open(args.log, "w") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
        )
        print(f"probe_optimized_server pid={process.pid} log={args.log}", flush=True)
        sys.exit(process.wait())


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
