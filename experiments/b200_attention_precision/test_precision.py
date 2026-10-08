"""Scope and receipt guards for projection-only precision interventions."""

import copy
import json

import pytest

from experiments.b200_attention_precision.worker import validate_native
from gleipnir.serving.fp4.attention import EXPECTED, ROWS, SHAPES, projection_identity
from gleipnir.serving.vllm.attention_precision import check_geometry


@pytest.mark.parametrize("identity", sorted(EXPECTED))
def test_geometry_covers_exactly_full_attention(identity):
    index, projection = identity
    check_geometry(
        f"model.layers.{index}.self_attn.{projection}",
        tuple(reversed(SHAPES[projection])),
    )


@pytest.mark.parametrize(
    "prefix",
    [
        "visual.layers.3.self_attn.qkv_proj",
        "model.layers.0.self_attn.qkv_proj",
        "model.layers.3.linear_attn.out_proj",
        "model.layers.3.mlp.down_proj",
    ],
)
def test_geometry_excludes_untargeted_modules(prefix):
    with pytest.raises(ValueError):
        check_geometry(prefix, (10240, 2560))


def native_receipt():
    return {
        "passed": True,
        "state": "completed",
        "checks": [
            {
                "precision": mode,
                "projection": p,
                "rows": m,
                "shape": list(SHAPES[p]),
                "passed": True,
                "finite": True,
                "zero_row_exact": True,
                "unchanged_rows_exact": True,
                "replay_changed": True,
                "relative_l2": 0.001,
                "replay_relative_l2": 0.001,
            }
            for mode in ("bf16", "fp8")
            for p in SHAPES
            for m in ROWS
        ],
    }


@pytest.mark.parametrize("mode", ["bf16", "fp8"])
def test_complete_native_receipt(mode):
    validate_native(native_receipt(), mode)


@pytest.mark.parametrize(
    "field,value",
    [
        ("relative_l2", float("nan")),
        ("replay_relative_l2", 0.02),
        ("unchanged_rows_exact", False),
        ("shape", [1, 2]),
    ],
)
def test_native_rejects_invalid_arithmetic_and_isolation(field, value):
    receipt = copy.deepcopy(native_receipt())
    receipt["checks"][0][field] = value
    with pytest.raises(ValueError):
        validate_native(receipt, "bf16")


def test_native_rejects_missing_case():
    receipt = native_receipt()
    receipt["checks"].pop(0)
    with pytest.raises(ValueError):
        validate_native(receipt, "bf16")


def test_vision_scope_remains_excluded():
    assert projection_identity("vision_model.layers.3.self_attn.qkv_proj") is None


@pytest.mark.parametrize("precision", ["bf16", "fp8"])
def test_candidate_command_preserves_non_target_recipe(precision, monkeypatch):
    from experiments.b200_attention_precision import run

    additional = {
        "serving_condition": {
            "gdn_projection_precision": "fp4",
            "attention_precision": "mxfp8",
            "attention_projection_precision": "fp4",
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
    command = run.make_command(parent, precision, "results/native.json")
    for flag in [
        "--model",
        "--max-model-len",
        "--max-num-seqs",
        "--max-num-batched-tokens",
        "--pooler-config",
        "--hf-overrides",
        "--scheduler-cls",
    ]:
        assert command[command.index(flag) + 1] == parent[parent.index(flag) + 1]
    old = json.loads(parent[parent.index("--additional-config") + 1])
    new = json.loads(command[command.index("--additional-config") + 1])
    for key, value in old["serving_condition"].items():
        if key != "attention_projection_precision":
            assert new["serving_condition"][key] == value
    assert new["serving_condition"]["attention_projection_precision"] == precision
    assert new["monitor_score"] == old["monitor_score"]
    assert new["runtime_migration"] == old["runtime_migration"]
