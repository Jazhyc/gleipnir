"""Serve a base model through the recorded optimized command of its Gleipnir release.

``--target 4b``: Qwen3.5-4B through Gleipnir 4B v0.4's command; ``27b``:
Qwen3.8-27B through Gleipnir 27B v0.2's. The release launcher's ``prepare``
builds the command and environment (runtime, source bindings, receipts, 32K
recorded context); only the model path changes to the base snapshot.
``--lens`` (4B) adds the eager Lens capture bridge, and ``--truncate`` swaps in
the worker that stops the decoder at mid-depth (after block 15 of 32, or 31
of 64) and returns the un-normalized residual stream; ``--probe-bank`` (with
truncation) also replaces the classify pooler with probe banks, ``--keep``
gives prompted rows' kept-span starts and ``--dump`` saves kept states.
The server runs in the foreground; the attempt log is under ``--log``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OVERLAY = "/tmp/gleipnir-vllm-lens-1.3.0"
SHARED_SOURCE = "experiments/probe_optimized_serving/truncate_worker.py"
# target: (recorded worker, truncated worker, sources bound into the compile identity)
WORKERS = {
    "4b": (
        "experiments.b200_gdn_fast.worker.FastGdnWorker",
        "experiments.probe_optimized_serving.truncate_worker.TruncatedFastGdnWorker",
        [SHARED_SOURCE],
    ),
    "27b": (
        "gleipnir.serving.quant_stack.QuantFullWorker",
        "experiments.probe_optimized_serving.truncate_worker_27b."
        "TruncatedQuantFullWorker",
        [SHARED_SOURCE, "experiments/probe_optimized_serving/truncate_worker_27b.py"],
    ),
}


def pristine_27b_sources() -> Path:
    """The 27B release image keeps the repository versions of the sources a 4B
    launch replaces with archived bytes. A fresh code sync has the same
    versions (no 4B launch may run between the sync and this one)."""
    from experiments.release_serving.targets import SHARED_DIAGNOSTIC_SOURCES

    root = Path(tempfile.mkdtemp(prefix="gleipnir-27b-sources-"))
    for relative in SHARED_DIAGNOSTIC_SOURCES:
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, root / relative)
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=sorted(WORKERS), default="4b")
    parser.add_argument("--snapshot", required=True, help="base model snapshot dir")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--lens", action="store_true")
    parser.add_argument("--truncate", action="store_true")
    parser.add_argument("--probe-bank", help="probe sweep dir")
    parser.add_argument("--epochs", default="2,4,8,16")
    parser.add_argument("--variants", default="fp8,raw")
    parser.add_argument("--select", help="comma-separated probe names (one epoch)")
    parser.add_argument("--keep", help="kept-span start table (prompted rows)")
    parser.add_argument("--dump", help="directory for kept states")
    parser.add_argument("--log", required=True)
    args = parser.parse_args()

    from experiments.release_serving import serve
    from experiments.release_serving.targets import RECORDED_CONTEXT, TARGETS
    from gleipnir.data.monitoring import file_hash

    if args.target == "27b":
        serve.PRISTINE = pristine_27b_sources()
    command, env = serve.prepare(
        TARGETS[args.target], "127.0.0.1", args.port, RECORDED_CONTEXT
    )
    command[command.index("--model") + 1] = args.snapshot
    index = command.index("--additional-config") + 1
    additional = json.loads(command[index])
    if "merged_model" in additional["serving_condition"]:
        # Part of the compile-cache key, as in bf16_launch.
        additional["serving_condition"]["merged_model"] = args.snapshot
    if args.truncate:
        recorded, truncated, sources = WORKERS[args.target]
        worker = command.index("--worker-cls") + 1
        if command[worker] != recorded:
            raise ValueError(f"unexpected selected worker {command[worker]}")
        command[worker] = truncated
        # Bind the worker sources into the compile identity, as the release
        # launcher does for its prefix-cached workers.
        for source in sources:
            additional["gleipnir_frost_fp4"][source] = file_hash(ROOT / source)
        env["GLEIPNIR_PROBE_RECEIPT"] = str(Path(args.log).with_suffix(".probe.json"))
        if args.probe_bank:
            env["GLEIPNIR_PROBE_BANK"] = str(ROOT / args.probe_bank)
            env["GLEIPNIR_PROBE_EPOCHS"] = args.epochs
            env["GLEIPNIR_PROBE_VARIANTS"] = args.variants
            if args.select:
                env["GLEIPNIR_PROBE_SELECT"] = args.select
            if args.keep:
                env["GLEIPNIR_PROBE_KEEP"] = str(ROOT / args.keep)
            if args.dump:
                env["GLEIPNIR_PROBE_DUMP"] = str(ROOT / args.dump)
    elif args.probe_bank:
        raise ValueError("the probe pooler requires --truncate")
    command[index] = json.dumps(additional, sort_keys=True)
    if args.lens:
        if args.target != "4b":
            raise ValueError("the Lens bridge was validated for the 4B only")
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
