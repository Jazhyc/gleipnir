"""Checksum-bound 20-update screen, conditional on compiled MLP pilot gains."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
from pathlib import Path

import yaml

from experiments.b200_nvidia_mxfp8.run import environment
from gleipnir.monitoring_systems_screen import sha256_file
from gleipnir.monitoring_training_command import training_command
from gleipnir.packed_benchmark import summarize

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = Path(
    "results/b200_bf16_fa4_accepted/flash_attention_4/causal_adapter/training_metadata.json"
)
REFERENCE_SHA256 = "185fa498f8ec31f07ae58a8213584c7a7f5b41738393ae75d97f91229a993364"


def accept_pilot(receipt: dict, *, minimum_improvement: float = 0.05) -> None:
    """Reject failed, partial, or eager-only speedups before expensive training."""
    if (
        receipt.get("status") != "complete"
        or receipt.get("variant") != "merged_compiled"
        or receipt.get("timing_baseline") != "compiled_peft"
        or not receipt.get("full_backward_included")
    ):
        raise ValueError("a complete compiled merged MLP pilot is required")
    rows = receipt["shapes"]
    if [r["tokens"] for r in rows] != [193, 4096, 16384]:
        raise ValueError("incomplete diagnostic shapes")
    for row in rows:
        errors = [row["output_relative_l2"], *row["gradient_relative_l2"]]
        if (
            row["finite"] is not True
            or len(row["gradient_relative_l2"]) != 7
            or not all(0 <= e <= 0.01 for e in errors)
            or not 0 <= row["row_isolation_relative_l2"] <= 1e-7
        ):
            raise ValueError("pilot arithmetic failed")
        samples = row["step_ms"]
        if set(samples) != {"baseline", "candidate"} or any(
            len(values) != 10 or not all(math.isfinite(v) and v > 0 for v in values)
            for values in samples.values()
        ):
            raise ValueError("incomplete or invalid pilot timings")
        gain = 1 - statistics.mean(samples["candidate"]) / statistics.mean(
            samples["baseline"]
        )
        if row["tokens"] >= 4096 and gain < minimum_improvement:
            raise ValueError("pilot lacks a five-percent compiled training gain")


def accept_integrated_pilot(
    receipt: dict, *, minimum_improvement: float = 0.05
) -> None:
    """Require complete FP4 autograd/oracle evidence and full-MLP graph gains."""
    if (
        receipt.get("status") != "complete"
        or receipt.get("variant") != "fp4_integrated"
        or receipt.get("timing_baseline") != "compiled_peft"
        or receipt.get("full_backward_included") is not True
    ):
        raise ValueError("complete registered FP4 MLP pilot required")
    rows = receipt["shapes"]
    if [r["tokens"] for r in rows] != [193, 4096, 16384]:
        raise ValueError("incomplete FP4 integration shapes")
    hardware = receipt.get("installation", {}).get("hardware_packing", False)
    fused = receipt.get("installation", {}).get("fused_descale", False)
    if fused and not hardware:
        raise ValueError("fused MLP must retain validated hardware packing")
    if hardware:
        packing = receipt.get("packing_validation", {})
        cases = packing.get("cases", [])
        if packing.get("status") != "complete" or [
            (c["rows"], c["width"]) for c in cases
        ] != [(m, k) for m in (193, 4096, 16384) for k in (2560, 9216, 18432)]:
            raise ValueError("complete hardware packing receipt required")
        if any(
            c.get("bitwise")
            != {"codes": True, "scales_including_padding": True, "row_inverse": True}
            or c.get("changed_input_bitwise") is not True
            or c.get("graph_copy_included") is not True
            for c in cases
        ):
            raise ValueError("hardware packing operand/replay mismatch")
    for row in rows:
        if fused:
            compared = row.get("four_way_graph_timing", {}).get("samples_ms", {})
            if set(compared) != {
                "baseline",
                "candidate",
                "reference_fp4",
                "hardware_fp4",
            } or any(
                len(v) != 10 or not all(math.isfinite(x) and x > 0 for x in v)
                for v in compared.values()
            ):
                raise ValueError("complete four-way fused MLP timings required")
            if any(
                compared[k] != row["graph_timing"]["samples_ms"][k]
                for k in ("baseline", "candidate")
            ):
                raise ValueError("fused MLP comparison timings drifted")
            if row["tokens"] >= 4096 and statistics.mean(compared["candidate"]) >= (
                statistics.mean(compared["hardware_fp4"])
            ):
                raise ValueError("fused MLP is not faster than hardware packing alone")
        if hardware and (
            row.get("reference_fp4_output_relative_l2") != 0
            or row.get("reference_fp4_gradient_relative_l2") != [0.0] * 7
        ):
            raise ValueError("hardware packing changed complete-MLP arithmetic")
        if (
            row["finite"] is not True
            or not 0 <= row["oracle_output_relative_l2"] <= 0.01
            or len(row["oracle_gradient_relative_l2"]) != 7
            or not all(0 <= e <= 0.02 for e in row["oracle_gradient_relative_l2"])
            or row["row_isolation_relative_l2"] != 0
            or row.get("graph_copy_included") is not True
            or len(row["changed_input_master_replay_relative_l2"]) != 8
            or not all(
                0 <= e <= 0.01 for e in row["changed_input_master_replay_relative_l2"]
            )
        ):
            raise ValueError("FP4 integration oracle/isolation/replay gate failed")
        samples = row["graph_timing"]["samples_ms"]
        if set(samples) != {"baseline", "candidate"} or any(
            len(v) != 10 or not all(math.isfinite(x) and x > 0 for x in v)
            for v in samples.values()
        ):
            raise ValueError("incomplete FP4 integration timings")
        gain = 1 - statistics.mean(samples["candidate"]) / statistics.mean(
            samples["baseline"]
        )
        if row["tokens"] >= 4096 and gain < minimum_improvement:
            raise ValueError("FP4 full MLP lacks five-percent graph gain")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--pilot-sha256", required=True)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--timing-authority")
    parser.add_argument("--warm-shapes", action="store_true")
    parser.add_argument("--profile-updates", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead"),
        default="reduce-overhead",
    )
    args = parser.parse_args()
    if not args.attempt.isalnum():
        raise ValueError("attempt must be alphanumeric")
    if sha256_file(args.pilot) != args.pilot_sha256:
        raise ValueError("pilot checksum drift")
    pilot = json.loads(args.pilot.read_text())
    fp4 = pilot.get("variant") == "fp4_integrated"
    if args.warm_shapes and (
        not fp4 or not args.timing_authority or args.compile_mode != "default"
    ):
        raise ValueError("warmed screen requires native FP4 timing in default mode")
    if args.profile_updates and not args.warm_shapes:
        raise ValueError("full-model profiling requires exact-shape warmup")
    if args.timing_authority is not None and (
        not fp4 or not args.timing_authority.strip()
    ):
        raise ValueError("timing-only authority requires a native FP4 pilot")
    hardware_packing = bool(pilot.get("installation", {}).get("hardware_packing"))
    fused_descale = bool(pilot.get("installation", {}).get("fused_descale"))
    (accept_integrated_pilot if fp4 else accept_pilot)(pilot)
    if sha256_file(ROOT / REFERENCE) != REFERENCE_SHA256:
        raise ValueError("FA4 historical control checksum drift")
    control = summarize(
        json.loads((ROOT / REFERENCE).read_text()), 10, accept_learning=True
    )
    baseline = json.loads(
        (ROOT / "results/b200_bf16_fa4_accepted/summary.json").read_text()
    )
    profile = ROOT / "src/gleipnir/configs/systems_screen/qwen35_4b_b200_bf16_fa4.yaml"
    recipe = yaml.safe_load(profile.read_text())["recipe"]
    output = ROOT / "results/b200_mlp_gemm" / args.attempt
    logs = ROOT / "logs/runpod/b200_mlp_gemm" / args.attempt
    output.mkdir(parents=True, exist_ok=False)
    logs.mkdir(parents=True, exist_ok=False)
    job = {
        **baseline["conditions"]["flash_attention_4"]["job"],
        **recipe,
        "job_name": "native-fp4-mlp-fa4" if fp4 else "merged-mlp-fa4",
        "max_steps": 20,
        "save_steps": 1000000,
        "expected_initial_master_sha256": baseline["conditions"]["flash_attention_4"][
            "initial_master_sha256"
        ],
        "output_dir": str(output),
        "causal_adapter_dir": str(output / "causal_adapter"),
        "model_dir": str(output / "model"),
        "hydra_log_dir": str(logs),
    }
    if fp4:
        job["packing_learning_gradient_tolerance"] = 0.05
        job["selective_torch_compile_mode"] = args.compile_mode
        job["native_fp4_hardware_packing"] = hardware_packing
        job["native_fp4_fused_descale"] = fused_descale
        if args.timing_authority:
            job["packing_timing_authority"] = args.timing_authority
    # MLP code changed; request fresh model gates instead of claiming startup reuse.
    job.pop("startup_validation_reference", None)
    job.pop("startup_validation_reference_sha256", None)
    if args.profile_updates:
        job["startup_validation_reference"] = str(
            ROOT
            / "results/b200_mlp_gemm/warmed03/causal_adapter/training_metadata.json"
        )
        job["startup_validation_reference_sha256"] = (
            "14ab15279bb8895cf32353117d5c1cf957ad45d2b0ca9d7205067db27d77edeb"
        )
    for path_key, hash_key in [
        ("student_rows", "student_rows_sha256"),
        ("soft_targets", "soft_targets_sha256"),
        ("selection_manifest", "selection_sha256"),
    ]:
        if sha256_file(Path(job[path_key])) != job[hash_key]:
            raise ValueError(f"input checksum drift: {path_key}")
    initial = ROOT / "results/fp4_row_aot_training/initial_adapter"
    for name, digest in baseline["initial_adapter_files"].items():
        if sha256_file(initial / name) != digest:
            raise ValueError(f"initial adapter drift: {name}")
    command = training_command(job) + [
        f"student.init_adapter={initial}",
        "++student.training.logging_steps=1",
    ]
    entry = command.index("experiments/deception_distillation/train_student_sft.py")
    command[entry : entry + 1] = [
        "-m",
        "experiments.b200_mlp_gemm.fp4_training_entry"
        if fp4
        else "experiments.b200_mlp_gemm.training_entry",
    ]
    if fp4:
        command += [
            f"++student.training.selective_torch_compile_mode={args.compile_mode}",
        ]
        if args.timing_authority:
            command.append("++student.training.native_fp4_mlp_timing=true")
        if args.profile_updates:
            command.append("++student.training.native_fp4_mlp_profile=true")
    cfg = yaml.safe_load((ROOT / "experiments/b200_mlp_gemm/config.yaml").read_text())
    env = environment(cfg)
    env.update(
        HF_HOME=str(ROOT / ".cache/huggingface"),
        HF_HUB_CACHE=str(ROOT / ".cache/huggingface/hub"),
        TORCHINDUCTOR_COMPILE_THREADS="16",
        MAX_JOBS="16",
        PYTHONUNBUFFERED="1",
        GLEIPNIR_FP4_RUNTIME_REPORT=str(output / "native_runtime.json"),
        GLEIPNIR_FP4_HARDWARE_PACKING="1" if hardware_packing else "0",
        GLEIPNIR_FP4_FUSED_DESCALE="1" if fused_descale else "0",
    )
    if args.warm_shapes:
        env.update(
            GLEIPNIR_FP4_WARM_REPORT=str(output / "shape_warmup.json"),
            GLEIPNIR_FP4_WARM_REFERENCE=str(ROOT / REFERENCE),
        )
    if args.profile_updates:
        env["GLEIPNIR_FP4_PROFILE_OUTPUT"] = str(output / "warmed_profile")
    sources = [
        Path("src/gleipnir/__init__.py"),
        Path("src/gleipnir/_compat.py"),
        *Path("experiments/b200_mlp_gemm").glob("*.py"),
        Path("src/gleipnir/kernels/mlp_gemm.py"),
        Path("src/gleipnir/kernels/fp4/cudnn_fp4_mlp.py"),
        Path("src/gleipnir/kernels/fp4/cudnn_fp4_epilogue.py"),
        Path("src/gleipnir/kernels/fp4/cudnn_fp4_gemm.py"),
        Path("src/gleipnir/kernels/fp4/nvfp4_pack.py"),
        Path("experiments/b200_mlp_gemm/README.md"),
        profile.relative_to(ROOT),
        Path("experiments/deception_distillation/train_student_sft.py"),
        *[
            Path("src/gleipnir") / name
            for name in (
                "training/packed.py",
                "__init__.py",
                "_compat.py",
                "packed_training_screen.py",
                "packed_benchmark.py",
                "attention_backends.py",
                "campaigns/training_command.py",
            )
        ],
    ]
    report = {
        "status": "starting",
        "job": job,
        "command": command,
        "pilot_sha256": args.pilot_sha256,
        "control_sha256": REFERENCE_SHA256,
        "mlp_intervention": "nvfp4_forward_dgrad" if fp4 else "merged_bf16",
        "hardware_packing": hardware_packing,
        "fused_descale": fused_descale,
        "candidate_compile_mode": args.compile_mode if fp4 else "default",
        "compile_mode_matches_control": not fp4 or args.compile_mode == "default",
        "timing_only": bool(args.timing_authority),
        "timing_authority": args.timing_authority,
        "exact_shape_warmup": args.warm_shapes,
        "profiled_updates": [11, 15, 20] if args.profile_updates else [],
        "instrumented_time_is_not_speed_result": args.profile_updates,
        "control": control,
        "cache_paths": {k: v for k, v in env.items() if "CACHE" in k},
        "source_sha256": {str(p): sha256_file(ROOT / p) for p in sources},
    }
    for p in sources:
        dest = output / "executed_sources" / p
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / p).read_bytes())
    summary = output / "summary.json"
    summary.write_text(json.dumps(report, indent=2) + "\n")
    with (logs / "training.log").open("x") as handle:
        code = subprocess.run(
            command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT
        ).returncode
    report.update(status="failed" if code else "training_complete", returncode=code)
    summary.write_text(json.dumps(report, indent=2) + "\n")
    if code:
        raise SystemExit(code)
    metadata = json.loads(
        (output / "causal_adapter/training_metadata.json").read_text()
    )
    candidate = summarize(
        metadata,
        10,
        accept_learning=not bool(args.timing_authority),
        accept_timing=bool(args.timing_authority),
    )
    for key in ("physical_contract", "initial_master_sha256"):
        if candidate[key] != control[key]:
            raise ValueError(f"physical contract mismatch: {key}")
    if args.warm_shapes:
        from experiments.b200_mlp_gemm.warmed_training import validate_warmed_receipt

        warm_receipt = json.loads((output / "shape_warmup.json").read_text())
        validate_warmed_receipt(warm_receipt, candidate)
        report["shape_warmup"] = warm_receipt
    gain = 1 - candidate["measured_mean_seconds"] / control["measured_mean_seconds"]
    report.update(
        status="complete",
        candidate=candidate,
        relative_improvement=None if args.profile_updates else gain,
        followup_supported=not args.profile_updates and gain >= 0.05,
        default_changed=False,
    )
    summary.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "complete",
                "relative_improvement": report["relative_improvement"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
