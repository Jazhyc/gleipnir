"""Keep native imports/context warm across queued FP4-output checker trials."""

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    args = parser.parse_args()
    args.queue.mkdir(parents=True, exist_ok=True)

    def state(status, **fields):
        p = args.queue / "state.json"
        temporary = p.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"pid": os.getpid(), "state": status, **fields}, indent=2) + "\n"
        )
        temporary.replace(p)

    state("initializing")
    import torch

    state("idle", runtime=torch.__version__)
    while not (args.queue / "stop").exists():
        requests = sorted(args.queue.glob("request_*.json"))
        if not requests:
            time.sleep(0.25)
            continue
        request = requests[0]
        spec = json.loads(request.read_text())
        running = request.with_suffix(".running")
        request.rename(running)
        state("running", output=spec["output"])
        for module in (
            "gleipnir.serving_fp4_swiglu_block_reference",
            "gleipnir.serving_fp4_swiglu_native_output",
            "experiments.b200_attention_gdn_serving.fp4_swiglu_native_output_compare",
        ):
            if module in sys.modules:
                importlib.reload(sys.modules[module])
        entry = importlib.import_module(
            "experiments.b200_attention_gdn_serving.fp4_swiglu_native_output_compare"
        )
        saved = sys.argv
        sys.argv = [entry.__file__, "--output", spec["output"]]
        try:
            entry.main()
        finally:
            sys.argv = saved
        receipt = json.loads(Path(spec["output"]).read_text())
        running.rename(running.with_suffix(".done"))
        if any(
            s in json.dumps(receipt).lower()
            for s in (
                "illegal instruction",
                "illegal memory access",
                "device-side assert",
            )
        ):
            state("poisoned_cuda", output=spec["output"])
            return
        state("idle", last_output=spec["output"], passed=receipt["passed"])
    state("stopped")


if __name__ == "__main__":
    main()
