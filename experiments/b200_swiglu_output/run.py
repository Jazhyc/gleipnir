"""One whole-row SwiGLU ablation on the selected FP8-attention monitor."""

import argparse
import asyncio
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np

from experiments.b200_attention_precision.run import condition
from experiments.b200_inference_benchmark.run import DATA, ROOT, sha, write
from experiments.b200_monitor_score.run import SERVING, trial
from experiments.b200_vllm031.id import bind_workload
from experiments.b200_vllm031.run import may_benchmark
from gleipnir.serving.fp4.swiglu_overhead_validation import (
    validate_native,
    validate_runtime,
)
from gleipnir.serving.reference import selected_serving_default

EXPERIMENT = Path(__file__).parent
SOURCE_PATHS = [
    "src/gleipnir/serving/fp4/swiglu_overhead.py",
    "src/gleipnir/serving/fp4/swiglu_overhead_integration.py",
    "src/gleipnir/serving/fp4/swiglu_pack.py",
    "experiments/b200_attention_precision/run.py",
    *[str(p.relative_to(ROOT)) for p in EXPERIMENT.glob("*.py")],
]


def make_command(parent: list[str], precision: str, native: str) -> list[str]:
    """Disable direct FP4 output, retaining all non-target precision and settings."""
    if precision != "whole_row":
        raise ValueError("only the predeclared whole-row SwiGLU candidate is supported")
    command = parent.copy()
    command[0] = sys.executable
    command[command.index("-m") + 1] = "experiments.b200_swiglu_output.server"
    command[command.index("--worker-cls") + 1] = (
        "experiments.b200_swiglu_output.worker.WholeRowSwiGluWorker"
    )
    index = command.index("--additional-config") + 1
    additional = json.loads(command[index])
    condition = additional["serving_condition"]
    if (
        condition["gdn_projection_precision"] != "fp4"
        or condition["attention_projection_precision"] != "fp8"
        or condition["attention_precision"] != "mxfp8"
    ):
        raise ValueError("parent is not the selected FP8 attention/FP4 SwiGLU recipe")
    condition["swiglu_direct_fp4_output"] = False
    condition["swiglu_output_ablation_validation"] = native
    sources = set(additional["gleipnir_frost_fp4"]) | set(SOURCE_PATHS)
    additional["gleipnir_frost_fp4"] = {p: sha(ROOT / p) for p in sorted(sources)}
    command[index] = json.dumps(additional, sort_keys=True)
    return command


async def canary(settings: dict, out: Path) -> None:
    rows = json.loads((DATA / "canary.json").read_text())
    baseline = json.loads(
        (ROOT / settings["candidate"] / "canary_predictions.json").read_text()
    )
    master = json.loads((ROOT / settings["master_canary"]).read_text())
    if (
        master["master_sha256"] != settings["master_sha256"]
        or [r["prompt_sha256"] for r in rows] != master["prompt_sha256"]
    ):
        raise ValueError("SwiGLU master canary identity changed")
    values, _ = await trial(rows, 4, settings)
    if [(r["id"], r["prompt_sha256"], r["prompt_tokens"]) for r in values] != [
        (r["id"], r["prompt_sha256"], r["prompt_tokens"]) for r in baseline
    ]:
        raise ValueError("SwiGLU canary input identity changed")
    scores = np.array([r["score"] for r in values])

    def drift(target):
        return {
            "mae": float(np.abs(scores - target).mean()),
            "correlation": float(np.corrcoef(scores, target)[0, 1]),
        }

    selected = drift(np.array([r["score"] for r in baseline]))
    master_drift = drift(np.array(master["adapter"]))
    receipt = {
        "passed": selected["mae"] <= 0.005 and selected["correlation"] >= 0.995,
        "mean_absolute_difference": selected["mae"],
        "correlation": selected["correlation"],
        "finite": bool(np.isfinite(scores).all())
        and all(np.isfinite(r["margin"]) for r in values),
        "adapter_effect": float(np.abs(scores - np.array(master["base"])).max()),
        "references": {"selected_fp8_attention": selected, "master_bf16": master_drift},
        "strict_master_passed": master_drift["mae"] <= 0.02
        and master_drift["correlation"] >= 0.99,
        "diagnostic_continuation_authorized": True,
    }
    write(out / "canary.json", receipt)
    write(out / "canary_predictions.json", values)
    if not may_benchmark(receipt, True):
        raise ValueError("SwiGLU finite/adapter effect guard failed")
    print("swiglu_canary", receipt, flush=True)


def scope_check(out: Path) -> None:
    write(
        out / "native_swiglu_whole_row.json",
        json.loads((SERVING / "native_swiglu_whole_row.json").read_text()),
    )
    loaded = json.loads((out / "loaded_precision.json").read_text())
    audit = json.loads((out / "native_swiglu_whole_row.json").read_text())
    condition = loaded["serving_condition"]
    validate_runtime(
        audit, loaded["worker_pid"], condition["swiglu_output_ablation_validation"]
    )
    if (
        audit["direct_output_calls"] != 0
        or audit["direct_fp4_output_enabled"] is not False
        or loaded["swiglu_native_output"]["enabled"] is not False
    ):
        raise ValueError("direct FP4 SwiGLU output is not disabled")
    for filename in (
        "native_attention_projections.json",
        "native_preparation.json",
        "native_gemm_tuning.json",
    ):
        receipt = json.loads((out / filename).read_text())
        if not receipt["passed"] or receipt["worker_pid"] != loaded["worker_pid"]:
            raise ValueError(f"retained execution audit failed: {filename}")
    if (
        loaded["mlp_projection_count"],
        loaded["gdn_projection_count"],
        loaded["attention_projection_count"],
    ) != (64, 48, 16):
        raise ValueError("SwiGLU retained precision coverage changed")


async def run(name: str, native_path: str) -> None:
    if not name or Path(name).name != name:
        raise ValueError("campaign name must be a stem")
    settings = json.loads((EXPERIMENT / "config.json").read_text())
    for path, expected in settings["files_sha256"].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f"SwiGLU frozen artifact changed: {path}")
    selection, command = selected_serving_default(ROOT)
    if selection["recipe_summary"] != settings["candidate"] + "/summary.json":
        raise ValueError("SwiGLU baseline differs from selected default")
    if {n: importlib.metadata.version(n) for n in selection["runtime"]} != selection[
        "runtime"
    ]:
        raise ValueError("SwiGLU runtime changed")
    native = json.loads((ROOT / native_path).read_text())
    validate_native(native)
    control = ROOT / settings["optimized_control"]
    workload = json.loads((control / "workload.json").read_text())
    reference = json.loads((control / "reference.json").read_text())
    optimized = bind_workload(
        workload, reference, json.loads((control / "repeat0.json").read_text())
    )
    default_id = bind_workload(
        workload,
        reference,
        json.loads((ROOT / settings["candidate"] / "id_predictions.json").read_text()),
    )
    merge = json.loads((control / "merged_artifact.json").read_text())
    model = Path(settings["merged_model"])
    if (
        json.loads((model / "merge_manifest.json").read_text()) != merge
        or merge["adapter_sha256"] != settings["serving_adapter_sha256"]
    ):
        raise ValueError("SwiGLU merged trained adapter identity changed")
    for path, expected in merge["files_sha256"].items():
        if sha(model / path) != expected:
            raise ValueError("SwiGLU merged weight changed")
    for field in ("master", "serving_adapter"):
        if sha(ROOT / settings[field]) != settings[field + "_sha256"]:
            raise ValueError("SwiGLU adapter source changed")
    out = ROOT / "results/b200_swiglu_output" / name
    out.mkdir(parents=True, exist_ok=False)
    write(out / "settings.json", settings)
    write(out / "native.json", native)
    write(out / "id_workload.json", workload)
    write(out / "id_reference.json", reference)
    stock = {
        "command": command,
        "runtime": selection["runtime"],
        "scheduler_binding": selection["scheduler_binding"],
    }
    await condition(
        name,
        "whole_row",
        native_path,
        settings,
        stock,
        workload,
        reference,
        optimized,
        default_id,
        result_group="b200_swiglu_output",
        command_builder=make_command,
        canary_check=canary,
        source_paths=SOURCE_PATHS,
        config_path=EXPERIMENT / "config.json",
        scope_check=scope_check,
        comparison_key="vs_selected_fp8_attention",
    )
    # Persist final actual dispatch after full development/ID passes.
    scope_check(out / "whole_row")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--native-receipt", required=True)
    args = parser.parse_args()
    asyncio.run(run(args.name, args.native_receipt))


if __name__ == "__main__":
    main()
