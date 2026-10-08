"""Campaign safety boundaries, staged recovery and frozen population contracts."""

import copy
import fcntl
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from gleipnir.campaigns.monitoring import __main__ as runner
from gleipnir.campaigns.monitoring.contract import Campaign
from gleipnir.campaigns.monitoring.evaluation import agreement, validate_workloads
from gleipnir.campaigns.monitoring.training import validate_completion
from gleipnir.data.monitoring import digest, write_json, write_rows

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "experiments/monitoring_campaign/config.yaml"


@pytest.fixture
def campaign(tmp_path):
    return Campaign.load(tmp_path, EXAMPLE)


def test_inventory_reports_all_missing_prerequisites(campaign):
    result = campaign.inventory()
    assert not result["passed"]
    for name in ("training", "teacher", "initial_weights", "id", "benchmark"):
        assert any(f"missing {name}:" in e for e in result["errors"])


def test_inventory_detects_file_drift_and_initializer_path_conflict(campaign):
    config = copy.deepcopy(campaign.config)
    config["inputs"]["teacher"]["path"] = "teacher.jsonl"
    (campaign.root / "teacher.jsonl").write_text("changed")
    config["model"]["initial_adapter"] = "another-initializer"
    result = replace(campaign, config=config).inventory()
    assert any("checksum mismatch teacher" in e for e in result["errors"])
    assert any("initial_adapter disagrees" in e for e in result["errors"])


@pytest.mark.parametrize("names", [["unknown"], ["id", "id"], []])
def test_unknown_or_repeated_evaluations_fail_before_inventory(tmp_path, names):
    config = yaml.safe_load(EXAMPLE.read_text())
    config["evaluations"] = names
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="registered"):
        Campaign.load(tmp_path, path)


def test_file_identity_and_tensor_identity_remain_distinct(campaign):
    job = campaign.job()
    assert (
        job["expected_initial_master_sha256"]
        == campaign.config["model"]["initial_tensor_sha256"]
    )
    assert (
        job["expected_initial_master_sha256"]
        != campaign.config["inputs"]["initial_weights"]["sha256"]
    )
    assert Path(job["startup_validation_reference"]).is_absolute()
    assert job["max_steps"] == -1 and job["num_train_epochs"] == 1.0


@pytest.fixture
def staged(tmp_path, monkeypatch):
    output = tmp_path / "output"
    checked = []
    ctx = SimpleNamespace(
        output=output,
        inventory=lambda **kw: {"passed": True},
        prepare=lambda: None,
        check=lambda: checked.append(True),
    )
    monkeypatch.setattr(runner, "products", lambda ctx, s: [output / f"{s}.artifact"])
    calls = []

    def launch(ctx, stage):
        calls.append(stage)
        (output / f"{stage}.artifact").write_text(stage)

    return ctx, launch, calls


def test_resume_skips_bound_completed_training_and_finishes_pipeline(staged):
    ctx, launch, calls = staged
    first = runner.execute(ctx, through="train", launch=launch)
    assert first["status"] == "paused_after_stage" and calls == ["train"]
    final = runner.execute(ctx, resume=True, launch=launch)
    assert final["status"] == "complete" and calls == list(runner.STAGES)
    runner.execute(ctx, resume=True, launch=launch)
    assert calls == list(runner.STAGES)


def test_existing_campaign_requires_explicit_resume(staged):
    ctx, launch, calls = staged
    runner.execute(ctx, through="train", launch=launch)
    with pytest.raises(ValueError, match="already started"):
        runner.execute(ctx, launch=launch)
    assert calls == ["train"]


def test_resume_rejects_changed_completed_weights(staged):
    ctx, launch, calls = staged
    runner.execute(ctx, through="train", launch=launch)
    (ctx.output / "train.artifact").write_text("changed weights")
    with pytest.raises(ValueError, match="output changed"):
        runner.execute(ctx, resume=True, launch=launch)
    assert calls == ["train"]


def test_failed_training_is_preserved_and_never_implicitly_repeated(staged):
    ctx, _, calls = staged

    def fail(ctx, stage):
        calls.append(stage)
        raise RuntimeError("nonfinite gradient")

    with pytest.raises(RuntimeError, match="nonfinite"):
        runner.execute(ctx, launch=fail)
    receipt = json.loads((ctx.output / "runner.json").read_text())
    assert receipt["stages"]["train"]["status"] == "failed"
    with pytest.raises(ValueError, match="unfinished/failed"):
        runner.execute(ctx, resume=True, launch=fail)
    assert calls == ["train"]


def test_missing_stage_products_cannot_be_marked_complete(staged):
    ctx, _, _ = staged
    with pytest.raises(ValueError, match="missing required"):
        runner.execute(ctx, launch=lambda *args: None)
    assert json.loads((ctx.output / "runner.json").read_text())["status"] == "failed"


def test_second_driver_cannot_launch_under_held_campaign_lock(staged):
    ctx, launch, calls = staged
    ctx.output.mkdir()
    with (ctx.output / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="another campaign driver"):
            runner.execute(ctx, launch=launch)
    assert not calls


def test_completed_coverage_uses_configured_row_count_and_rejects_duplicate(
    monkeypatch,
):
    from gleipnir.campaigns.monitoring import training

    monkeypatch.setattr(training, "validate_training_metadata", lambda *a, **kw: None)
    ctx = SimpleNamespace(
        config={
            "logical_batch_size": 32,
            "training_rows": 35,
            "expected_training_tokens": 200,
            "expected_steps": 2,
            "model": {"initial_tensor_sha256": "fingerprint"},
        },
        job=lambda: {},
        profile=lambda: {},
    )
    meta = {
        "quantization": {"full_bf16_lora": {}},
        "adaptive_microbatching": {
            "logical_batch_sizes": [32, 3],
            "records": [
                {
                    "update": 1,
                    "logical_indices": list(range(32)),
                    "tokens": 100,
                    "padded_tokens": 100,
                },
                {
                    "update": 2,
                    "logical_indices": [0, 1, 2],
                    "tokens": 100,
                    "padded_tokens": 100,
                },
            ],
        },
        "train_metrics": {"epoch": 1.0},
        "optimizer_step_timing": {"durations_seconds": [1, 2]},
    }
    validate_completion(meta, ctx)
    meta["quantization"]["full_bf16_lora"]["native_fp4_mlp"] = {"base_forward": "nvfp4"}
    with pytest.raises(ValueError, match="MLP training precision"):
        validate_completion(meta, ctx)
    meta["quantization"]["full_bf16_lora"].pop("native_fp4_mlp")
    meta["adaptive_microbatching"]["records"][1]["logical_indices"] = [0, 1, 1]
    with pytest.raises(ValueError, match="missing or repeated"):
        validate_completion(meta, ctx)


def test_workloads_reject_identity_metadata_prompt_and_holdout_overlap(tmp_path):
    canonical = [
        {
            "id": "held-out",
            "prompt": "source",
            "metadata": {
                "source_dataset": "source",
                "ground_truth": 0,
                "trajectory_sha256": "held-out-hash",
            },
        }
    ]
    work = [
        {
            "id": "held-out",
            "prompt": "rendered",
            "prompt_sha256": digest("rendered"),
            "prompt_tokens": 10,
            "dataset": "source",
            "label": 0,
            "source_prompt_sha256": digest("source"),
        }
    ]
    write_rows(tmp_path / "id", canonical)
    write_json(tmp_path / "id_workload", work)
    write_rows(tmp_path / "training", [{"trajectory_sha256": "train-hash"}])
    ctx = SimpleNamespace(
        splits=("id",),
        input=lambda name: tmp_path / name,
        config={"evaluation": {"populations": {"id": 1}}},
    )
    assert len(validate_workloads(ctx)["id"]) == 1
    bad = copy.deepcopy(work)
    bad[0]["label"] = 1
    write_json(tmp_path / "id_workload", bad)
    with pytest.raises(ValueError, match="label drift"):
        validate_workloads(ctx)
    write_json(tmp_path / "id_workload", work)
    write_rows(tmp_path / "training", [{"trajectory_sha256": "held-out-hash"}])
    with pytest.raises(ValueError, match="overlap"):
        validate_workloads(ctx)


def test_new_adapter_gates_do_not_inherit_old_finite_waiver(campaign):
    limits = campaign.config["parity"]
    reference, base = [0.1, 0.2, 0.8, 0.9], [0.4] * 4
    assert agreement([0.101, 0.202, 0.803, 0.904], reference, base, limits)["passed"]
    assert not agreement([0.3, 0.4, 0.6, 0.7], reference, base, limits)["passed"]
    assert not agreement(reference, reference, reference, limits)["passed"]
    assert not agreement([float("nan"), 0.2, 0.8, 0.9], reference, base, limits)[
        "passed"
    ]


@pytest.mark.parametrize(
    "stage", ["train", "master-reference", "merged-reference", "evaluate"]
)
def test_worker_uses_isolated_runtime_and_records_launch(campaign, monkeypatch, stage):
    captured = {}
    monkeypatch.setattr(Campaign, "check", lambda self: {})
    monkeypatch.setattr(
        runner,
        "training_environment",
        lambda *a, **kw: {"PYTHONPATH": "/pinned-kernels"},
    )

    def start(command, **kwargs):
        captured.update(command=command, **kwargs)
        return SimpleNamespace(pid=123, wait=lambda: 0)

    monkeypatch.setattr(runner.subprocess, "Popen", start)
    runner.worker(campaign, stage)
    key = "serving_python" if stage == "evaluate" else "training_python"
    assert captured["command"][0] == campaign.config[key]
    assert ("experiments.b200_vllm031.runtime" in captured["command"]) == (
        stage == "evaluate"
    )
    assert captured["env"]["HF_HUB_OFFLINE"] == "1"
    assert captured["env"]["GLEIPNIR_CAMPAIGN_DRIVER_PID"]
    if stage != "evaluate":
        assert "/pinned-kernels" in captured["env"]["PYTHONPATH"].split(":")
    assert captured["start_new_session"]
    receipt = json.loads(
        (campaign.output / "stage_launches" / f"{stage}.json").read_text()
    )
    assert receipt["pid"] == 123 and receipt["command"] == captured["command"]
    assert Path(receipt["log"]).parent == campaign.logs / "stages"
    assert Path(receipt["log"]) != campaign.logs / "train.log"


def test_bf16_profile_keeps_fa4_and_routes_runtime_without_fp4(campaign, monkeypatch):
    config = copy.deepcopy(campaign.config)
    config["profile"] = "qwen35_4b_b200_bf16_fa4"
    bf16 = replace(campaign, config=config)
    fp4_job, bf16_job = campaign.job(), bf16.job()
    changed = {
        key
        for key in fp4_job.keys() | bf16_job.keys()
        if fp4_job.get(key) != bf16_job.get(key)
    }
    assert changed == {
        "native_fp4_mlp",
        "native_fp4_mlp_parity_policy",
        "startup_validation_reference_sha256",
        "packing_learning_gradient_tolerance",
    }
    assert bf16_job["packed_attention_backend"] == "flash_attention_4"
    assert not bf16_job.get("native_fp4_mlp", False)
    captured = {}
    monkeypatch.setattr(Campaign, "check", lambda self: {})

    def environment(*args, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(runner, "training_environment", environment)
    monkeypatch.setattr(
        runner.subprocess,
        "Popen",
        lambda *a, **kw: SimpleNamespace(pid=123, wait=lambda: 0),
    )
    runner.worker(bf16, "train")
    assert captured["native_fp4_mlp"] is False


def test_bf16_campaign_accepts_only_named_validated_profiles(tmp_path):
    config = yaml.safe_load(EXAMPLE.read_text())
    path = tmp_path / "config.yaml"
    config["profile"] = "qwen35_4b_b200_bf16_fa4"
    path.write_text(yaml.safe_dump(config))
    assert (
        Campaign.load(tmp_path, path).profile()["recipe"]["packed_attention_backend"]
        == "flash_attention_4"
    )
    config["profile"] = "qwen35_4b_b200_fast"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="validated"):
        Campaign.load(tmp_path, path)


def test_missing_runtime_or_wrong_checkout_fails_before_gpu(campaign):
    result = campaign.inventory(runtime=True)
    assert any("GPU run must use" in error for error in result["errors"])
    assert any(
        "missing executable training_python" in error for error in result["errors"]
    )


def test_diagnostic_preserves_failed_mae_and_rejects_nonfinite_or_ranking_drift(
    campaign,
):
    from gleipnir.campaigns.monitoring.evaluation import require_evaluation_gate

    reference, base = [0.1, 0.2, 0.8, 0.9], [0.4] * 4
    failed = agreement(
        [0.13, 0.23, 0.83, 0.93], reference, base, campaign.config["parity"]
    )
    gate = {**failed, "versus_merged_bf16": dict(failed)}
    assert not gate["passed"] and gate["finite"]
    with pytest.raises(ValueError, match="agreement failed"):
        require_evaluation_gate(gate, diagnostic=False)
    require_evaluation_gate(gate, diagnostic=True)
    assert not gate["passed"] and not gate["versus_merged_bf16"]["passed"]
    for key, value in [("finite", False), ("correlation", 0.5), ("adapter_effect", 0)]:
        broken = copy.deepcopy(gate)
        broken["versus_merged_bf16"][key] = value
        with pytest.raises(ValueError, match="agreement failed"):
            require_evaluation_gate(broken, diagnostic=True)
