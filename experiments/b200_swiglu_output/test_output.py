"""Guard the narrow output ablation and reject direct-output execution."""

import asyncio
import hashlib
import json
from types import SimpleNamespace

import pytest


def test_command_changes_only_output_policy(monkeypatch):
    from experiments.b200_swiglu_output import run

    additional = {
        "serving_condition": {
            "gdn_projection_precision": "fp4",
            "attention_projection_precision": "fp8",
            "attention_precision": "mxfp8",
            "fp4_preparation": "combined",
        },
        "gleipnir_frost_fp4": {},
        "monitor_score": {"token_ids": [15, 16]},
    }
    parent = [
        "python",
        "-m",
        "old",
        "--worker-cls",
        "old",
        "--quantization",
        "gleipnir_frost_attention_precision",
        "--additional-config",
        json.dumps(additional),
        "--model",
        "merged",
        "--max-model-len",
        "32768",
        "--max-num-seqs",
        "128",
        "--pooler-config",
        "LAST",
        "--scheduler-cls",
        "stock",
    ]
    monkeypatch.setattr(run, "sha", lambda path: "fixture")
    command = run.make_command(parent, "whole_row", "results/native.json")
    assert command[5:7] == parent[5:7]
    assert command[9:] == parent[9:]
    changed = json.loads(command[8])
    for k, v in additional["serving_condition"].items():
        assert changed["serving_condition"][k] == v
    assert changed["serving_condition"]["swiglu_direct_fp4_output"] is False
    assert (
        changed["serving_condition"]["swiglu_output_ablation_validation"]
        == "results/native.json"
    )
    assert changed["monitor_score"] == additional["monitor_score"]
    assert json.loads(parent[8]) == additional
    with pytest.raises(ValueError):
        run.make_command(parent, "fp8", "results/native.json")


def test_worker_replaces_output_and_rejects_direct_dispatch(tmp_path, monkeypatch):
    import torch

    from experiments.b200_attention_gdn_serving.test_swiglu_overhead import receipt
    from experiments.b200_swiglu_output import worker as module
    from gleipnir.serving.fp4 import swiglu_native_output_integration as direct
    from gleipnir.serving.fp4 import swiglu_overhead_integration as whole_row

    native = receipt()
    native.update(
        gpu="fixture",
        sources={},
        kernel={"generated_sha256": "whole", "upstream_sha256": "upstream"},
    )
    for r in native["results"]:
        r["changed_row_effect"] = True
    results = tmp_path / "results"
    results.mkdir()
    (results / "native.json").write_text(json.dumps(native))
    (results / "previous.json").write_text(json.dumps({"kernel": native["kernel"]}))
    (results / "direct.json").write_text("validated direct receipt")
    audit_path = results / "b200_attention_gdn_serving/native_swiglu_output.json"
    audit_path.parent.mkdir()
    audit_path.write_text(
        json.dumps(
            {
                "overhead_reference_sha256": hashlib.sha256(
                    (results / "previous.json").read_bytes()
                ).hexdigest()
            }
        )
    )
    plan = SimpleNamespace(pad=False, plan=object(), compile_count=1)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "fixture")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(direct, "_PLAN", SimpleNamespace(overhead=plan))
    monkeypatch.setattr(direct, "_AUDIT", None)
    monkeypatch.setattr(module, "write", lambda *args: None)
    callback = {}

    def install(model, received_plan, audit):
        assert received_plan is plan
        callback["audit"] = audit
        return {"mlp_count": 32, "minimum_rows": 1536}

    monkeypatch.setattr(whole_row, "install", install)
    monkeypatch.setattr(
        module.AttentionPrecisionWorker, "load_model", lambda self, **kwargs: None
    )
    candidate = module.WholeRowSwiGluWorker.__new__(module.WholeRowSwiGluWorker)
    candidate.vllm_config = SimpleNamespace(
        additional_config={
            "serving_condition": {
                "swiglu_direct_fp4_output": False,
                "swiglu_output_ablation_validation": "results/native.json",
                "swiglu_overhead_reference": "results/previous.json",
                "swiglu_native_output_validation": "results/direct.json",
            }
        }
    )
    candidate.model_runner = SimpleNamespace(get_model=lambda: object())
    candidate.precision = {"swiglu_native_output": {"generated_sha256": "direct"}}
    candidate.audit_serving_state = lambda: None
    candidate.load_model()
    assert candidate.precision["swiglu_native_output"]["enabled"] is False
    assert candidate.precision["swiglu_whole_row"]["mlp_count"] == 32
    for layer in range(32):
        callback["audit"](layer, 32768, "fused")
    with pytest.raises(ValueError, match="direct FP4.*executed"):
        direct._AUDIT(0, 32768, "fused")


@pytest.mark.parametrize("mode", ["finite_drift", "nonfinite", "no_effect"])
def test_canary_shared_guard_and_diagnostic_receipt(tmp_path, monkeypatch, mode):
    from experiments.b200_swiglu_output import run

    rows = [
        {
            "id": str(i),
            "prompt_sha256": str(i),
            "prompt_tokens": 10,
            "score": s,
            "margin": s,
        }
        for i, s in enumerate([0.2, 0.8])
    ]
    base = [0.0, 0.0]
    observed = [dict(r, score=r["score"] + 0.02) for r in rows]
    if mode == "nonfinite":
        observed[0]["score"] = float("nan")
    if mode == "no_effect":
        base = [r["score"] for r in observed]
    (tmp_path / "canary.json").write_text(json.dumps(rows))
    (tmp_path / "baseline").mkdir()
    (tmp_path / "baseline/canary_predictions.json").write_text(json.dumps(rows))
    (tmp_path / "master.json").write_text(
        json.dumps(
            {
                "master_sha256": "fixture",
                "prompt_sha256": [r["prompt_sha256"] for r in rows],
                "base": base,
                "adapter": [r["score"] for r in rows],
            }
        )
    )

    async def trial(*args):
        return observed, 1.0

    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setattr(run, "DATA", tmp_path)
    monkeypatch.setattr(run, "trial", trial)
    settings = {
        "candidate": "baseline",
        "master_canary": "master.json",
        "master_sha256": "fixture",
    }
    if mode == "finite_drift":
        asyncio.run(run.canary(settings, tmp_path / "out"))
        receipt = json.loads((tmp_path / "out/canary.json").read_text())
        assert not receipt["passed"]
        assert run.may_benchmark(receipt, True)
        assert not run.may_benchmark(receipt, False)
    else:
        with pytest.raises(ValueError, match="finite/adapter effect"):
            asyncio.run(run.canary(settings, tmp_path / "out"))
