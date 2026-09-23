"""Audit screen artifacts and combine numerical and native timing evidence."""

import json
from pathlib import Path

from experiments.local_inference.core import write_json
from gleipnir.qwen35_adapter_rebase import sha256_file


def main() -> None:
    root = Path("results/int4_calibration")
    screen = json.loads((root / "screen/result.json").read_text())
    manifest = json.loads((root / "capture/manifest.json").read_text())
    native = json.loads((root / "native_recovered/result.json").read_text())
    fast = json.loads((root / "native_fht/result.json").read_text())
    assert len(screen["projections"]) == 6 and len(native["rows"]) == 12
    assert len(fast["rows"]) == 2
    identities = [(r["projection"], r["recipe"]["family"]) for r in native["rows"]]
    assert len(set(identities)) == 12
    cal = {
        r["original_trajectory_sha256"]
        for r in manifest["rows"]
        if r["split"] == "calibration"
    }
    held = {
        r["original_trajectory_sha256"]
        for r in manifest["rows"]
        if r["split"] == "heldout"
    }
    assert len(cal) == 8 and len(held) == 4 and not cal & held
    for projection in screen["projections"]:
        selection = json.loads(
            (root / "screen" / f"{projection['projection']}_selection.json").read_text()
        )
        for family, entry in projection["selected"].items():
            best = min(
                (t for t in selection["trials"] if t["recipe"]["family"] == family),
                key=lambda t: t["calibration"]["relative_l2"],
            )
            assert entry["recipe"] == best["recipe"]
    timings = []
    for row in native["rows"] + fast["rows"]:
        base = next(
            r["gpu_ms"]
            for r in native["rows"]
            if r["projection"] == row["projection"] and r["recipe"]["family"] == "bf16"
        )
        timings.append({**row, "speedup_vs_bf16": base / row["gpu_ms"]})
    result = {
        "macro_heldout_relative_l2": screen["macro_heldout_relative_l2"],
        "macro_calibration_relative_l2": {
            family: sum(
                p["selected"][family]["calibration"]["relative_l2"]
                for p in screen["projections"]
            )
            / 6
            for family in screen["projections"][0]["selected"]
        },
        "fallback_projections": screen["fallback_projections"],
        "timings": timings,
        "sha256": {
            str(p): sha256_file(p)
            for p in [
                root / "capture/manifest.json",
                root / "screen/result.json",
                root / "native/result.json",
                root / "native_recovered/result.json",
                root / "native_fht/result.json",
            ]
        },
        "code_sha256": {
            str(p): sha256_file(p)
            for p in Path("experiments/int4_calibration").glob("*.py")
        },
        "note": "Calibration-only choices, four short held-out trajectories; "
        "numeric dequantization screen is not native serving or AUROC. Native "
        "timing covers only layer 0 with real calibration activations. "
        "One original smooth timing recovered without repetition; corrected "
        "validation checks native rounding exactly and reports simulation differences. "
        "Fast-Hadamard is a distinct implementation timed in a separate process. "
        "Uncontrolled temperatures and short windows; no variance estimate.",
    }
    write_json(root / "comparison.json", result)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("timings", "sha256", "code_sha256")
            },
            indent=2,
        )
    )
    for row in timings:
        print(
            row["projection"],
            row["recipe"],
            round(row["gpu_ms"], 4),
            round(row["speedup_vs_bf16"], 3),
        )


if __name__ == "__main__":
    main()
