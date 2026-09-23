"""Record local GPU telemetry; this is not an agent monitoring heartbeat."""

from __future__ import annotations

import datetime
import json
import subprocess
import time

from experiments.local_inference.core import ROOT


def main() -> None:
    fields = (
        "temperature.gpu,clocks.current.sm,clocks.current.memory,memory.used,"
        "utilization.gpu,power.draw,clocks_throttle_reasons.sw_thermal_slowdown,"
        "clocks_throttle_reasons.hw_thermal_slowdown"
    )
    with (ROOT / "baseline/gpu_telemetry.jsonl").open("a") as handle:
        while True:
            status = json.loads((ROOT / "baseline/status.json").read_text())
            root_status = json.loads((ROOT / "status.json").read_text())
            if status["state"] == "complete" or root_status["state"] == "failed":
                break
            values = (
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        f"--query-gpu={fields}",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                )
                .strip()
                .split(", ")
            )
            sample = {
                "utc": datetime.datetime.now(datetime.UTC).isoformat(),
                "status": status,
                "gpu": dict(zip(fields.split(","), values, strict=True)),
            }
            handle.write(json.dumps(sample) + "\n")
            handle.flush()
            time.sleep(10)


if __name__ == "__main__":
    main()
