import json
from pathlib import Path

from gleipnir.monitoring_systems_screen import (
    load_config,
    make_jobs,
    resolve_paths,
    summarize_screen,
    validate_training_metadata,
)

CONFIG_PATH = Path("experiments/monitoring_inner_mixer_compile/config.json")
CONTROL_NAME = "linear-shell-compile-control"
CANDIDATE_NAME = "inner-linear-mixer-compile-candidate"
CONTROL_POLICY = "full_attention_and_linear_shell"
CANDIDATE_POLICY = "full_attention_and_linear_mixer_segments"
OPTIMIZER_STEPS = 8
EXPECTED_ALL_LAYERS = list(range(32))
EXPECTED_CHECKPOINTED_LAYERS = [
    index for index in EXPECTED_ALL_LAYERS if (index + 1) % 4 != 0
]


def metadata(policy: str, rate: float) -> dict:
    candidate = policy == CANDIDATE_POLICY
    return {
        "gradient_checkpointing": True,
        "gradient_checkpointing_policy": "linear_attention_only",
        "gradient_checkpointing_layer_indices": None,
        "checkpointed_layer_indices": EXPECTED_CHECKPOINTED_LAYERS,
        "selective_torch_compile": {
            "policy": policy,
            "compiled_layer_indices": EXPECTED_ALL_LAYERS,
            "disabled_linear_attention_layer_indices": (
                [] if candidate else EXPECTED_CHECKPOINTED_LAYERS
            ),
            "disabled_linear_attention_kernel_layer_indices": (
                EXPECTED_CHECKPOINTED_LAYERS if candidate else []
            ),
            "disabled_linear_attention_kernel_components": (
                ["causal_conv1d_fn", "chunk_gated_delta_rule", "norm.forward"]
                if candidate
                else []
            ),
            "backend": "inductor",
            "mode": "default",
            "dynamic": True,
            "global_torch_compile": False,
            "dynamo_counters": {"stats": {"unique_graphs": 7 if candidate else 3}},
            "canary": None,
        },
        "materialized_training_inputs": {"completion": False, "direct": True},
        "training_batch": {
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 32,
            "effective_batch_size": 32,
        },
        "flash_linear_attention": {
            "available": True,
            "required": True,
            "version": "0.5.2",
            "backend_dispatch_disabled": True,
            "triton_version": "3.7.1",
        },
        "gated_delta_kernel_modules": ["fla.ops.gated_delta_rule.chunk"],
        "causal_conv1d": {
            "available": True,
            "required": True,
            "version": "1.6.2.post1",
            "kernel_modules": ["causal_conv1d.causal_conv1d_interface"],
        },
        "training_state": {"global_step": OPTIMIZER_STEPS},
        "train_metrics": {
            "train_samples_per_second": rate,
            "train_runtime": 256 / rate,
        },
        "optimizer_step_timing": {
            "recorded_steps": OPTIMIZER_STEPS,
            "steady_mean_seconds": 30.0,
        },
        "optimization": {"optimizer": "adamw", "trainer_optim": "adamw_torch"},
        "direct_logits_mode": "selected_positions",
        "peak_cuda_memory_allocated_bytes": 28 * 2**30,
    }


def test_jobs_are_matched_eight_step_conditions(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    paths = resolve_paths(config, data_dir=tmp_path / "data", result_dir=tmp_path)
    jobs = make_jobs(config, paths, "selection-digest")
    assert [job["job_name"] for job in jobs] == [CONTROL_NAME, CANDIDATE_NAME]
    assert {job["max_steps"] for job in jobs} == {8}
    assert {
        job["selective_torch_compile_policy"] for job in jobs
    } == {CONTROL_POLICY, CANDIDATE_POLICY}
    assert jobs[1]["selective_torch_compile_policy"] == CANDIDATE_POLICY


def test_summary_rejects_sub_one_percent_gain(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    paths = resolve_paths(config, data_dir=tmp_path / "data", result_dir=tmp_path)
    jobs = make_jobs(config, paths, "selection-digest")
    jobs[0]["training_metadata"] = metadata(CONTROL_POLICY, 1.0)
    jobs[1]["training_metadata"] = metadata(CANDIDATE_POLICY, 1.009)
    assert summarize_screen(config, jobs)["selected_value"] == CONTROL_POLICY
    jobs[1]["training_metadata"]["train_metrics"]["train_samples_per_second"] = 1.01
    assert summarize_screen(config, jobs)["selected_value"] == CANDIDATE_POLICY


def test_preflight_requires_compile_canary(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    paths = resolve_paths(config, data_dir=tmp_path / "data", result_dir=tmp_path)
    job = make_jobs(config, paths, "selection-digest")[1]
    training_metadata = metadata(CANDIDATE_POLICY, 1.0)
    training_metadata["training_state"]["global_step"] = 1
    training_metadata["optimizer_step_timing"]["recorded_steps"] = 1
    try:
        validate_training_metadata(
            training_metadata,
            config,
            job,
            expected_steps=1,
            require_canary=True,
        )
    except ValueError as error:
        assert "canary" in str(error)
    else:
        raise AssertionError("missing compile canary was accepted")


def test_config_freezes_eight_steps() -> None:
    config = json.loads(CONFIG_PATH.read_text())
    assert config["systems_screen_schema"] == 1
    assert config["recipe"]["max_steps"] == 8
    assert config["comparison"]["minimum_relative_improvement"] == 0.01
