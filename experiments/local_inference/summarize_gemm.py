"""Recover completed down timings from the interrupted screen without rerunning."""

import json
from pathlib import Path

from experiments.local_inference.core import ROOT, write_json


def recover_down(log: str, reference: dict) -> dict:
    """Require unique timing rows and exact agreement with recovery validation."""
    rows = [
        json.loads(line[5:]) for line in log.splitlines() if line.startswith("down {")
    ]
    rows += reference["measurements"]
    names = [row["candidate"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate timing; refusing to select between repeats")
    if names.count("torch_default") != 1:
        raise ValueError("Exactly one default reference is required")
    for row in rows:
        for key, value in reference["validation"][row["candidate"]].items():
            if row[key] != value:
                raise ValueError(f"Validation mismatch: {row['candidate']} {key}")
    base = next(r["gpu_ms_per_call"] for r in rows if r["candidate"] == "torch_default")
    for row in rows:
        row["speedup_vs_default"] = base / row["gpu_ms_per_call"]
    return {**reference, "measurements": rows}


def main() -> None:
    initial = ROOT / "gemm_bench_bf16"
    recovery = ROOT / "gemm_bench_down_reference"
    log = Path("logs/local/local_inference/gemm_bench_bf16.log")
    original = json.loads((initial / "result.json").read_text())
    reference = json.loads((recovery / "result.json").read_text())["shapes"][0]
    down = recover_down(log.read_text(), reference)
    if len(original["shapes"]) != 1 or len(down["measurements"]) != 11:
        raise ValueError("Unexpected partial-run layout")
    comparison = {
        "shapes": original["shapes"] + [down],
        "sources": [
            str(initial / "result.json"),
            str(log),
            str(recovery / "result.json"),
        ],
        "caveat": "Initial run timed ten down alternatives but skipped the default "
        "after its numerical guard failed, then crashed computing speedup. "
        "Only the missing default timing was recovered in another process. "
        "Validation matches exactly. No completed timing was repeated. "
        "Thermals and clocks were uncontrolled; small differences are inconclusive.",
    }
    write_json(initial / "comparison.json", comparison)
    for shape in comparison["shapes"]:
        best = min(shape["measurements"], key=lambda r: r["gpu_ms_per_call"])
        print(shape["projection"], best["candidate"], best["speedup_vs_default"])


if __name__ == "__main__":
    main()
