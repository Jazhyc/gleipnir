"""State dtype, gate interpretation, receipt admission and launch identity."""

import json

import pytest
import torch

from experiments.b200_gdn_state.run import screen, state_command
from gleipnir.serving.gdn.state import make_forward, validate_native


def test_bf16_native_state_and_fp32_exp_gates_preserve_inputs():
    calls = []

    def kernel(**kwargs):
        calls.append(kwargs)
        return (
            (kwargs["v"], kwargs["output_state"])
            if kwargs["output_final_state"]
            else kwargs["v"]
        )

    q = torch.ones(1, 3, 2, 4, dtype=torch.bfloat16)
    g = torch.full((1, 3, 2), -0.1)
    beta = torch.full((1, 3, 2), 0.731)
    state = torch.randn(1, 2, 4, 4)
    before = state.clone()
    forward = make_forward(kernel, lambda x: x)
    output, final = forward(q, q, q, g, beta, state, True, torch.tensor([0, 3]))
    call = calls[-1]
    assert torch.equal(call["g"], g.squeeze(0).exp())
    assert call["g"].dtype == call["beta"].dtype == torch.float32
    assert call["initial_state"].dtype == final.dtype == torch.bfloat16
    assert torch.equal(state, before) and output.shape == q.shape
    assert forward(q, q, q, g, beta, state, False, torch.tensor([0, 3]))[1] is None
    rounded = make_forward(kernel, lambda x: x, round_gates=True)
    rounded(q, q, q, g, beta, state, True, torch.tensor([0, 3]))
    assert torch.equal(calls[-1]["g"], g.squeeze(0).exp().bfloat16().float())
    assert torch.equal(calls[-1]["beta"], beta.squeeze(0).bfloat16().float())
    with pytest.raises(ValueError, match="FP32 or BF16"):
        forward(q, q, q, g, beta, state.half(), True)


def test_incomplete_or_precision_changed_native_receipt_is_rejected():
    receipt = {
        "passed": True,
        "intervention": "flashinfer_gdn_bf16_state",
        "state_dtype": "bfloat16",
        "gate_dtype": "float32",
        "accumulation_dtype": "float32",
        "relative_l2_limit": 0.03,
        "sources": {"kernel.py": "hash"},
        "checks": [
            {"lengths": v, "passed": True}
            for v in ([1], [17], [129], [1, 127, 513], [4096], [8192, 8192], [32768])
        ],
        "oracle_passed": True,
        "continuation_passed": True,
        "isolation_passed": True,
        "graph_replay_passed": True,
    }
    validate_native(receipt)
    for key, value in (
        ("gate_dtype", "bfloat16"),
        ("accumulation_dtype", "bfloat16"),
        ("graph_replay_passed", False),
        ("relative_l2_limit", 0.04),
    ):
        with pytest.raises(ValueError, match="native admission"):
            validate_native({**receipt, key: value})


def test_launch_keeps_endpoint_and_arithmetic_and_binds_state_sources():
    additional = {
        "serving_condition": {"attention_precision": "mxfp8"},
        "gleipnir_frost_fp4": {},
        "monitor_score": {"token_ids": [15, 16]},
    }
    original = [
        "python",
        "-m",
        "experiments.b200_mutation_analysis.server",
        "--worker-cls",
        "old.worker",
        "--quantization",
        "gleipnir_frost_attention_fp4",
        "--runner",
        "pooling",
        "--additional-config",
        json.dumps(additional),
    ]
    command = state_command(
        original,
        {},
        {"src/gleipnir/serving/gdn/state.py": "state_hash"},
        validation="results/native.json",
    )
    assert "--mamba-ssm-cache-dtype" not in original
    assert command[command.index("--mamba-ssm-cache-dtype") + 1] == "bfloat16"
    assert (
        command[command.index("--quantization") + 1]
        == original[original.index("--quantization") + 1]
    )
    assert command[command.index("--runner") + 1] == "pooling"
    config = json.loads(command[command.index("--additional-config") + 1])
    assert config["serving_condition"]["gdn_state_validation"] == "results/native.json"
    assert (
        config["gleipnir_frost_fp4"]["src/gleipnir/serving/gdn/state.py"]
        == "state_hash"
    )
    with pytest.raises(ValueError, match="already overrides"):
        state_command(command, {}, {}, validation="results/native.json")


def test_screen_requires_speed_latency_and_ranking_together():
    def report(speed, latency):
        return {
            "trials": [
                {"concurrency": 128, "prompt_tokens_per_second": speed},
                {
                    "concurrency": 1,
                    "latency": {"p50_seconds": latency, "p95_seconds": latency * 2},
                },
            ]
        }

    ranking = {"auroc_delta": {"macro": 0.0005, "pooled": -0.0005}}
    reference = report(100, 1)
    assert screen(report(102, 1), reference, ranking)["passed"]
    assert not screen(report(100.5, 1), reference, ranking)["passed"]
    assert not screen(report(102, 1.03), reference, ranking)["passed"]
    assert not screen(
        report(102, 1), reference, {"auroc_delta": {"macro": 0.0011, "pooled": 0}}
    )["passed"]


def test_exited_startup_recovery_preserves_receipts_and_refuses_live_gpu(
    tmp_path, monkeypatch
):
    from experiments.b200_gdn_state import run

    serving = tmp_path / "results/serving"
    serving.mkdir(parents=True)
    log = tmp_path / "logs/runpod/b200_attention_gdn_serving/server.log"
    log.parent.mkdir(parents=True)
    log.write_text("failed startup traceback\n")
    metadata = {
        "pid": 99999999,
        "command": ["python", "--worker-cls", run.STATE_WORKER],
    }
    (serving / "server.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setattr(run, "SERVING", serving)
    monkeypatch.setattr(run.subprocess, "check_output", lambda *a, **k: "12345\n")
    with pytest.raises(ValueError, match="GPU workers remain"):
        run.archive_exited_candidate("failed01")
    assert (serving / "server.json").exists() and log.exists()
    monkeypatch.setattr(run.subprocess, "check_output", lambda *a, **k: "")
    run.archive_exited_candidate("failed01")
    assert not (serving / "server.json").exists()
    assert (
        json.loads((serving / "failed01_candidate_server_exited.json").read_text())[
            "status"
        ]
        == "exited"
    )
    assert (
        log.with_name("failed01_candidate_server.log").read_text()
        == "failed startup traceback\n"
    )


def test_failed_trial_retires_candidate_without_restarting_reference(
    tmp_path, monkeypatch
):
    import asyncio

    from experiments.b200_gdn_state import run

    serving = tmp_path / "results/serving"
    serving.mkdir(parents=True)
    parent = tmp_path / "parent.json"
    parent.write_text(json.dumps({"pid": 99999999, "command": ["python"]}))
    calls = []

    async def failed_measure(name, **kwargs):
        calls.append(name)
        out = tmp_path / "results/b200_gdn_state" / name
        out.mkdir(parents=True)
        (out / "parent_server.json").write_text(parent.read_text())
        (serving / "server.json").write_text(
            json.dumps(
                {
                    "pid": 99999999,
                    "command": ["python", "--worker-cls", run.STATE_WORKER],
                }
            )
        )
        raise RuntimeError("candidate failed")

    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setattr(run, "SERVING", serving)
    monkeypatch.setattr(run, "resume_environment", lambda parent: {})
    monkeypatch.setattr(run, "measure", failed_measure)
    monkeypatch.setattr(
        run, "archive_exited_candidate", lambda name: (serving / "server.json").unlink()
    )
    with pytest.raises(RuntimeError, match="candidate failed"):
        asyncio.run(run.run("failed01", retired_parent=parent))
    assert calls == ["failed01"]
    receipt = json.loads(
        (tmp_path / "results/b200_gdn_state/failed01/recovery.json").read_text()
    )
    assert receipt["candidate_retired"] and not receipt["reference_restored"]
