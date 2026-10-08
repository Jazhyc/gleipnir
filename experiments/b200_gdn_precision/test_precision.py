"""Guard GDN geometry, finite native receipts and projection-only command scope."""

import asyncio
import copy
import json

import pytest

from gleipnir.serving.gdn_precision import (
    EXPECTED,
    ROWS,
    SHAPES,
    check_geometry,
    validate_native,
)


@pytest.mark.parametrize("identity", sorted(EXPECTED))
def test_all_48_gdn_geometries(identity):
    index, projection = identity
    check_geometry(
        f"model.layers.{index}.linear_attn.{projection}",
        tuple(reversed(SHAPES[projection])),
    )


@pytest.mark.parametrize(
    "prefix",
    [
        "model.layers.3.linear_attn.out_proj",
        "model.layers.0.linear_attn.in_proj_a",
        "model.layers.0.mlp.down_proj",
        "model.layers.3.self_attn.qkv_proj",
        "visual.layers.0.linear_attn.out_proj",
        "model.layers.32.linear_attn.out_proj",
    ],
)
def test_excludes_untargeted_modules(prefix):
    with pytest.raises(ValueError, match="unsupported GDN"):
        check_geometry(prefix, (2560, 4096))


def receipt():
    return {
        "passed": True,
        "state": "completed",
        "checks": [
            {
                "projection": p,
                "rows": m,
                "precision": "fp8",
                "shape": list(SHAPES[p]),
                "passed": True,
                "finite": True,
                "zero_row_exact": True,
                "unchanged_rows_exact": True,
                "replay_changed": True,
                "relative_l2": 0.001,
                "replay_relative_l2": 0.001,
            }
            for p in SHAPES
            for m in ROWS
        ],
    }


def test_complete_native_receipt():
    validate_native(receipt())


@pytest.mark.parametrize(
    "field,value",
    [
        ("relative_l2", float("nan")),
        ("replay_relative_l2", 0.02),
        ("unchanged_rows_exact", False),
        ("shape", [1, 2]),
        ("precision", "fp4"),
    ],
)
def test_rejects_bad_native_receipt(field, value):
    value_receipt = copy.deepcopy(receipt())
    value_receipt["checks"][0][field] = value
    with pytest.raises(ValueError, match="failed GDN"):
        validate_native(value_receipt)


def test_rejects_duplicate_native_case():
    value = receipt()
    value["checks"][-1] = value["checks"][0]
    with pytest.raises(ValueError, match="incomplete GDN"):
        validate_native(value)


def test_command_changes_only_gdn_scope(monkeypatch):
    from experiments.b200_gdn_precision import run

    additional = {
        "serving_condition": {
            "gdn_projection_precision": "fp4",
            "attention_projection_precision": "fp8",
            "attention_precision": "mxfp8",
            "fp4_preparation": "combined",
        },
        "gleipnir_frost_fp4": {"src/gleipnir/serving/vllm/frost_fp4.py": "old"},
        "monitor_score": {"token_ids": [15, 16]},
        "runtime_migration": {"vllm": "0.31.0", "gdn_cp": "auto"},
    }
    parent = [
        "python",
        "-m",
        "old_server",
        "--worker-cls",
        "old_worker",
        "--quantization",
        "old_quantizer",
        "--additional-config",
        json.dumps(additional),
        "--model",
        "merged",
        "--max-model-len",
        "32768",
        "--max-num-seqs",
        "128",
        "--max-num-batched-tokens",
        "32768",
        "--pooler-config",
        "LAST",
        "--hf-overrides",
        "causal",
        "--scheduler-cls",
        "stock",
    ]
    monkeypatch.setattr(run, "sha", lambda path: "fixture_hash")
    command = run.make_command(parent, "fp8", "results/native.json")
    assert command[9:] == parent[9:]
    new = json.loads(command[8])
    for key, value in additional["serving_condition"].items():
        if key != "gdn_projection_precision":
            assert new["serving_condition"][key] == value
    assert new["serving_condition"]["gdn_projection_precision"] == "fp8"
    assert new["monitor_score"] == additional["monitor_score"]
    assert new["runtime_migration"] == additional["runtime_migration"]
    assert json.loads(parent[8]) == additional
    with pytest.raises(ValueError):
        run.make_command(parent, "bf16", "results/native.json")


@pytest.mark.parametrize("mode", ["finite_drift", "nonfinite", "no_effect"])
def test_canary_shared_guard_and_diagnostic_receipt(tmp_path, monkeypatch, mode):
    from experiments.b200_gdn_precision import run

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
