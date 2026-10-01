"""Run bounded native FP4 forward diagnostics or precision trajectories."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from experiments.b200_fouroversix.run import make_job, verify_inputs
from experiments.fp4_stability.memory_recipe import (
    apply_memory_recipe,
    validate_memory_recipe,
)
from experiments.tool_trajectory_monitoring.run_distillation_train import (
    training_command,
)
from gleipnir.monitoring_systems_screen import gpu_environment, sha256_file
from gleipnir.qwen35_fast_training import (
    DEFAULT_CAUSAL_CONV1D_TARGET,
    DEFAULT_FLA_TARGET,
    DEFAULT_TRITON_TARGET,
    causal_conv1d_environment,
    fla_environment,
    triton_environment,
)

ROOT = Path(__file__).resolve().parents[2]


def validate_config(config: dict) -> None:
    """Fail before GPU model loading for unsupported or unbounded campaigns."""
    validate_memory_recipe(config)

    def check_layers(indices):
        return (
            isinstance(indices, list)
            and bool(indices)
            and all(type(i) is int and 0 <= i < 32 and i % 4 != 3 for i in indices)
            and indices == sorted(set(indices))
        )

    if config.get("flashqla_layer_indices") is not None and not check_layers(
        config["flashqla_layer_indices"]
    ):
        raise ValueError("invalid FlashQLA decoder-layer selection")
    sweep = config.get("flashqla_layer_sweep")
    if sweep is not None and (
        not config.get("diagnostics_only", False)
        or not isinstance(sweep, list)
        or not sweep
        or not all(check_layers(indices) for indices in sweep)
        or config.get("flashqla_layer_indices") is not None
    ):
        raise ValueError("invalid diagnostic-only FlashQLA layer sweep")
    if (
        sweep is not None or config.get("flashqla_layer_indices") is not None
    ) and config.get("gated_delta_backend") != "flashqla":
        raise ValueError("layer selection requires FlashQLA")
    from gleipnir.flashqla_training import BOUNDARY_POLICIES

    policy = config.get("gated_delta_boundary_policy", "bf16")
    if policy not in BOUNDARY_POLICIES or (
        policy != "bf16" and not config.get("gated_delta_bf16_boundary", False)
    ):
        raise ValueError("unknown or inactive GDN boundary policy")
    if config.get("gated_delta_backend", "fla") not in {"fla", "flashqla", "fla_bf16"}:
        raise ValueError("unknown gated-delta backend")
    if (
        config.get("flashqla_auto_cp", False)
        and config.get("gated_delta_backend", "fla") != "flashqla"
    ):
        raise ValueError("automatic FlashQLA partitioning requires FlashQLA")
    if config.get("gated_delta_backend", "fla") == "flashqla" and not config.get(
        "flashqla_canary"
    ):
        raise ValueError("FlashQLA requires a recorded isolated canary")
    if config["steps"] not in {1, 10}:
        raise ValueError("retain one preflight update or ten matched updates")
    if (
        config["diagnostics_only"]
        and config["steps"] != 1
        and config.get("diagnostic_selection") != "matched_320"
    ):
        raise ValueError("forward diagnostics must use the global preflight selection")
    if not config["conditions"] or any(
        name not in {"nf4", "bf16", "fouroversix"} for name in config["conditions"]
    ):
        raise ValueError("unknown or empty precision conditions")
    if config["backward_mode"] not in {"fp4", "dequantized_bf16"}:
        raise ValueError("unknown backward precision")
    if config["compile_backend"] not in {"inductor", "aot_eager", "eager"}:
        raise ValueError("unknown diagnostic compiler backend")
    if config.get("activation_selector", "strict") not in {"strict", "fp16"}:
        raise ValueError("unknown activation selector")
    if config.get("activation_selector", "strict") == "fp16" and not (
        config["conditions"] == ["fouroversix"]
        and config.get("row_scaled_activations", False)
        and config.get("fused_row_scaling", False)
        and not config.get("fused_activation_packing", False)
        and not config.get("capture_native_operands", False)
        and config["backward_mode"] == "dequantized_bf16"
    ):
        raise ValueError(
            "FP16 selector requires unobserved native fused rows and BF16 backward"
        )
    if config.get("compiler_visible_native", False) and not (
        config["conditions"] == ["fouroversix"]
        and config["backward_mode"] == "dequantized_bf16"
        and not config.get("capture_native_operands", False)
    ):
        raise ValueError(
            "compiler-visible native requires unobserved FP4 and BF16 backward"
        )
    if bool(config.get("initial_adapter")) != bool(
        config.get("expected_initial_master_sha256")
    ):
        raise ValueError("initial adapter requires its expected master hash")
    if config.get("timing_repeats", 0) not in {0, 3} or (
        config.get("timing_repeats", 0)
        and (config["steps"] != 10 or config["diagnostics_only"])
    ):
        raise ValueError(
            "timing benchmark requires ten steps and three measured replays"
        )
    if config.get("profile_batch") is not None and (
        config["steps"] != 10
        or config["diagnostics_only"]
        or not 1 <= config["profile_batch"] <= 10
    ):
        raise ValueError("profiling requires one of ten warmed training batches")
    if config.get("gradient_validation", "per_tensor") not in {
        "per_tensor",
        "clip_norm",
    }:
        raise ValueError("unknown adapter gradient validation mode")
    if config.get("fused_row_scaling", False) and not config.get(
        "row_scaled_activations", False
    ):
        raise ValueError("fused row scaling requires per-token activations")
    if config.get("fused_activation_packing", False) and not (
        config.get("row_scaled_activations", False)
        and config.get("fused_row_scaling", False)
        and config["backward_mode"] == "dequantized_bf16"
        and config["conditions"] == ["fouroversix"]
        and not config.get("capture_native_operands", False)
    ):
        raise ValueError(
            "fused packing requires unobserved native per-token BF16 backward"
        )
    if config.get("share_gate_up_activations", False) and not (
        config.get("row_scaled_activations", False)
        and config.get("fused_row_scaling", False)
        and config["backward_mode"] == "dequantized_bf16"
        and config["conditions"] == ["fouroversix"]
        and not config.get("capture_native_operands", False)
    ):
        raise ValueError(
            "shared packing requires unobserved native per-token BF16 backward"
        )


def campaign_stages(config: dict, source: dict, global_longest: dict) -> list[dict]:
    """Require an actual global-longest update before every ten-update condition."""
    stages = []
    for precision in config["conditions"]:
        if config["steps"] == 10 and not config.get("diagnostics_only", False):
            stages.append(
                dict(
                    name=f"{precision}-global-preflight",
                    precision=precision,
                    source=global_longest,
                    steps=1,
                )
            )
        stages.append(
            dict(
                name=precision,
                precision=precision,
                source=source,
                steps=config["steps"],
            )
        )
    return stages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    validate_config(config)
    flashqla_canary = None
    if config.get("gated_delta_backend", "fla") == "flashqla":
        flashqla_canary = ROOT / config["flashqla_canary"]
        canary = json.loads(flashqla_canary.read_text())
        if canary["status"] != "passed" or canary["auto_cp"] != config.get(
            "flashqla_auto_cp", False
        ):
            raise ValueError("FlashQLA isolated canary failed or policy mismatched")
        if config.get("gated_delta_bf16_boundary", False) != canary.get(
            "bf16_boundary", False
        ):
            raise ValueError("FlashQLA canary BF16 boundary mismatch")
        if config.get("gated_delta_boundary_policy", "bf16") != canary.get(
            "boundary_policy", "bf16"
        ):
            raise ValueError("FlashQLA canary boundary policy mismatch")
    output = ROOT / config["output"]
    output.mkdir(parents=True, exist_ok=False)
    logs = ROOT / config["logs"]
    logs.mkdir(parents=True, exist_ok=True)
    source_path = config["preflight_jobs"] if config["steps"] == 1 else config["jobs"]
    jobs = [json.loads(line) for line in (ROOT / source_path).read_text().splitlines()]
    source = (
        jobs[0]
        if config["steps"] == 1
        else next(job for job in jobs if job["job_name"] == config["condition_source"])
    )
    verify_inputs(source)
    global_jobs = [
        json.loads(line)
        for line in (ROOT / config["preflight_jobs"]).read_text().splitlines()
        if line
    ]
    if len(global_jobs) != 1:
        raise ValueError("expected the frozen global-longest-32 selection")
    verify_inputs(global_jobs[0])
    environment = triton_environment(
        DEFAULT_TRITON_TARGET,
        causal_conv1d_environment(
            DEFAULT_CAUSAL_CONV1D_TARGET, fla_environment(DEFAULT_FLA_TARGET)
        ),
    )
    environment["PYTHONPATH"] = (
        f"{ROOT / config['kernel_target']}:{environment['PYTHONPATH']}"
    )
    environment["FLA_DISABLE_BACKEND_DISPATCH"] = "1"
    if config.get("gated_delta_backend", "fla") == "flashqla":
        from gleipnir.flashqla_training import FLASHQLA_TARGET

        environment["PYTHONPATH"] = (
            f"{ROOT / FLASHQLA_TARGET}:{environment['PYTHONPATH']}"
        )
        environment["TILELANG_CACHE_DIR"] = str(
            ROOT
            / config.get("flashqla_compiler_cache", ".cache/training/flashqla/tilelang")
        )
    environment["OMP_NUM_THREADS"] = "4"
    environment["TORCHINDUCTOR_EMULATE_PRECISION_CASTS"] = (
        "1" if config.get("emulate_precision_casts", False) else "0"
    )
    cache = ROOT / config["compiler_cache"]
    cache.mkdir(parents=True, exist_ok=True)
    environment = gpu_environment(environment, 0, cache)
    contract = {
        "config": config,
        "source_job": source,
        "global_longest_job": global_jobs[0],
        "inputs_verified": True,
        "inductor_emulate_precision_casts": config.get(
            "emulate_precision_casts", False
        ),
        "source_revision": os.environ.get("GLEIPNIR_COMMIT"),
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in [
                ROOT / "src/gleipnir/fouroversix_training.py",
                ROOT / "src/gleipnir/fp4_compiler_diagnostic.py",
                ROOT / "src/gleipnir/precision_training_screen.py",
                ROOT / "src/gleipnir/fp4_performance.py",
                ROOT / "src/gleipnir/fp4_row_kernels.py",
                ROOT / "src/gleipnir/fp4_quantization_kernels.py",
                ROOT / "src/gleipnir/fp4_fast_selector.py",
                ROOT / "src/gleipnir/fp4_compiler_ops.py",
                ROOT / "src/gleipnir/fp4_memory.py",
                ROOT / "src/gleipnir/flashqla_training.py",
                ROOT / "src/gleipnir/fp32_projection.py",
                ROOT / "experiments/fp4_stability/memory_recipe.py",
                ROOT / "experiments/fp4_stability/row_kernel_canary.py",
                ROOT / "experiments/fp4_stability/packing_kernel_canary.py",
                ROOT / "experiments/fp4_stability/shared_activation_canary.py",
                ROOT / "experiments/fp4_stability/fast_selector_canary.py",
                ROOT / "experiments/fp4_stability/compiler_op_canary.py",
                ROOT / "experiments/fp4_stability/reference_offload_canary.py",
                ROOT / "experiments/b200_fouroversix/kernel_canary.py",
                ROOT / "experiments/deception_distillation/train_student_sft.py",
                Path(__file__),
                args.config.resolve(),
            ]
        },
    }
    if config.get("initial_adapter"):
        initial_adapter = ROOT / config["initial_adapter"]
        contract["initial_adapter_sha256"] = {
            filename: sha256_file(initial_adapter / filename)
            for filename in ["adapter_config.json", "adapter_model.safetensors"]
        }
    if flashqla_canary is not None:
        contract["flashqla_canary_sha256"] = sha256_file(flashqla_canary)
    (output / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    status = {"status": "running", "stages": []}

    def publish() -> None:
        (output / "status.json").write_text(json.dumps(status, indent=2) + "\n")

    publish()
    try:
        if config.get("reference_weights_on_cpu", False):
            offload_command = [
                sys.executable,
                "experiments/fp4_stability/reference_offload_canary.py",
                "--output",
                str(output / "reference_offload_canary.json"),
                "--activation-selector",
                config.get("activation_selector", "strict"),
            ]
            if config.get("compiler_visible_native", False):
                offload_command.append("--compiler-visible-native")
            with (logs / "reference-offload-canary.log").open("w") as handle:
                subprocess.run(
                    offload_command,
                    cwd=ROOT,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        if config.get("compiler_visible_native", False):
            with (logs / "compiler-op-canary.log").open("w") as handle:
                subprocess.run(
                    [
                        sys.executable,
                        "experiments/fp4_stability/compiler_op_canary.py",
                        "--output",
                        str(output / "compiler_op_canary.json"),
                        "--activation-selector",
                        config.get("activation_selector", "strict"),
                    ],
                    cwd=ROOT,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        if config.get("activation_selector", "strict") == "fp16":
            with (logs / "fast-selector-canary.log").open("w") as handle:
                subprocess.run(
                    [
                        sys.executable,
                        "experiments/fp4_stability/fast_selector_canary.py",
                        "--output",
                        str(output / "fast_selector_canary.json"),
                    ],
                    cwd=ROOT,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        if config.get("share_gate_up_activations", False):
            with (logs / "shared-activation-canary.log").open("w") as handle:
                subprocess.run(
                    [
                        sys.executable,
                        "experiments/fp4_stability/shared_activation_canary.py",
                        "--output",
                        str(output / "shared_activation_canary.json"),
                    ],
                    cwd=ROOT,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        if config.get("fused_activation_packing", False):
            with (logs / "packing-kernel-canary.log").open("w") as handle:
                subprocess.run(
                    [
                        sys.executable,
                        "experiments/fp4_stability/packing_kernel_canary.py",
                        "--implementation",
                        "tiled",
                        "--output",
                        str(output / "packing_kernel_canary.json"),
                    ],
                    cwd=ROOT,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        if config.get("fused_row_scaling", False):
            with (logs / "row-kernel-canary.log").open("w") as handle:
                subprocess.run(
                    [
                        sys.executable,
                        "experiments/fp4_stability/row_kernel_canary.py",
                        "--output",
                        str(output / "row_kernel_canary.json"),
                    ],
                    cwd=ROOT,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        canary_command = [
            sys.executable,
            "experiments/b200_fouroversix/kernel_canary.py",
            "--output",
            str(output / "kernel_canary.json"),
            "--backward-mode",
            config["backward_mode"],
            "--activation-selector",
            config.get("activation_selector", "strict"),
        ]
        if config.get("row_scaled_activations", False):
            canary_command.append("--row-scaled-activations")
        if config.get("fused_row_scaling", False):
            canary_command.append("--fused-row-scaling")
        if config.get("fused_activation_packing", False):
            canary_command.append("--fused-activation-packing")
        with (logs / "kernel-canary.log").open("w") as handle:
            subprocess.run(
                canary_command,
                cwd=ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )
        for specification in campaign_stages(config, source, global_jobs[0]):
            precision = specification["precision"]
            stage_steps = specification["steps"]
            destination = output / specification["name"]
            job = make_job(specification["source"], destination, precision, stage_steps)
            job = apply_memory_recipe(job, config)
            command = training_command(job) + [
                f"++student.quantization.mlp_precision={precision}",
                f"++student.quantization.fp4_backward_mode={config['backward_mode']}",
                "++student.quantization.fp4_row_scaled_activations="
                f"{str(config.get('row_scaled_activations', False)).lower()}",
                "++student.training.adapter_init_seed=0",
                "++student.training.precision_screen.enabled=true",
                f"++student.training.precision_screen.output_dir={destination}",
                f"++student.training.precision_screen.steps={stage_steps}",
                f"++student.training.precision_screen.diagnostics_only={str(config['diagnostics_only']).lower()}",
                "++student.training.precision_screen.capture_native_operands="
                f"{str(config.get('capture_native_operands', False)).lower()}",
                "++student.training.precision_screen.eager_rmsnorm_interfaces="
                f"{str(config.get('eager_rmsnorm_interfaces', False)).lower()}",
                "++student.training.precision_screen.eager_mlp_activation_interfaces="
                f"{str(config.get('eager_mlp_activation_interfaces', False)).lower()}",
                f"student.training.selective_torch_compile_backend={config['compile_backend']}",
                "++student.quantization.fp4_fused_row_scaling="
                f"{str(config.get('fused_row_scaling', False)).lower()}",
                "++student.quantization.fp4_fused_activation_packing="
                f"{str(config.get('fused_activation_packing', False)).lower()}",
                "++student.quantization.fp4_share_gate_up_activations="
                f"{str(config.get('share_gate_up_activations', False)).lower()}",
                "++student.quantization.fp4_activation_selector="
                f"{config.get('activation_selector', 'strict')}",
                "++student.quantization.fp4_compiler_visible_native="
                f"{str(config.get('compiler_visible_native', False)).lower()}",
                "++student.training.precision_screen.gradient_validation="
                f"{config.get('gradient_validation', 'per_tensor')}",
                "++student.training.precision_screen.reference_weights_on_cpu="
                f"{str(config.get('reference_weights_on_cpu', False)).lower()}",
                "++student.training.precision_screen.gated_delta_backend="
                f"{config.get('gated_delta_backend', 'fla')}",
                "++student.training.precision_screen.flashqla_auto_cp="
                f"{str(config.get('flashqla_auto_cp', False)).lower()}",
                "++student.training.precision_screen.gated_delta_bf16_boundary="
                f"{str(config.get('gated_delta_bf16_boundary', False)).lower()}",
                "++student.training.precision_screen.gated_delta_boundary_policy="
                f"{config.get('gated_delta_boundary_policy', 'bf16')}",
                "++student.training.precision_screen.fp32_lm_head="
                f"{str(config.get('fp32_lm_head', False)).lower()}",
            ]
            if config.get("compile_policy"):
                command.append(
                    "student.training.selective_torch_compile_policy="
                    f"{config['compile_policy']}"
                )
            for key in ["flashqla_layer_indices", "flashqla_layer_sweep"]:
                if config.get(key) is not None:
                    command.append(
                        f"++student.training.precision_screen.{key}="
                        + json.dumps(config[key], separators=(",", ":"))
                    )
            if stage_steps == 1:
                command.append("student.training.warmup_ratio=0.0")
            elif config.get("timing_repeats", 0):
                command.append(
                    "++student.training.precision_screen.timing_repeats="
                    f"{config['timing_repeats']}"
                )
            if stage_steps == 10 and config.get("profile_batch") is not None:
                command.append(
                    "++student.training.precision_screen.profile_batch="
                    f"{config['profile_batch']}"
                )
            if config.get("initial_adapter"):
                command.extend(
                    [
                        f"student.init_adapter={ROOT / config['initial_adapter']}",
                        "++student.training.precision_screen.expected_initial_master_sha256="
                        f"{config['expected_initial_master_sha256']}",
                    ]
                )
            stage = {
                "name": specification["name"],
                "precision": precision,
                "steps": stage_steps,
                "selection_manifest": specification["source"]["selection_manifest"],
                "command": command,
                "execution_job": job,
                "status": "running",
                "started_unix": time.time(),
            }
            status["stages"].append(stage)
            publish()
            print(f"starting_stage={specification['name']}", flush=True)
            with (logs / f"{specification['name']}.log").open("w") as handle:
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            stage.update(
                returncode=result.returncode,
                ended_unix=time.time(),
                status="complete" if result.returncode == 0 else "failed",
            )
            publish()
            if result.returncode:
                raise RuntimeError(f"{specification['name']} failed; inspect {logs}")
        status["status"] = "complete"
    except Exception as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        publish()


if __name__ == "__main__":
    main()
