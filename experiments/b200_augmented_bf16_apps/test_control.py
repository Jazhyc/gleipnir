"""The resident APPS pass cannot silently score a different model or precision."""

from copy import deepcopy

import pytest

from experiments.b200_augmented_bf16_apps.run import validate_resident


def evidence() -> tuple[dict, dict]:
    server = {
        "pid": 123,
        "status": "ready",
        "serving_precision": "bf16",
        "command": ["python", "-m", "bf16_server"],
        "adapter_sha256": "fixed",
    }
    native = {
        "passed": True,
        "quantization": None,
        "projection_counts": {"attention": 16, "gdn": 48, "mlp": 64},
        "attention_calls": [
            {
                "query_dtype": "torch.bfloat16",
                "cache_dtype": "torch.bfloat16",
                "causal": True,
            }
            for _ in range(8)
        ],
    }
    return server, native


@pytest.mark.parametrize(
    "field,value",
    [
        ("pid", 456),
        ("adapter_sha256", "other"),
        ("status", "retired"),
        ("serving_precision", "optimized"),
    ],
)
def test_changed_resident_identity_is_rejected(field, value):
    expected, native = evidence()
    server = {**expected, field: value}
    with pytest.raises(ValueError, match="identity"):
        validate_resident(server, expected, native)


def test_native_attention_must_be_actual_bf16_and_complete():
    server, native = evidence()
    validate_resident(server, server, native)
    bad = deepcopy(native)
    bad["attention_calls"][0]["query_dtype"] = "torch.float8_e4m3fn"
    with pytest.raises(ValueError, match="native audit"):
        validate_resident(server, server, bad)
    native["attention_calls"].pop()
    with pytest.raises(ValueError, match="native audit"):
        validate_resident(server, server, native)


def test_matching_eager_or_quantized_receipt_is_still_rejected():
    server, native = evidence()
    server["command"].append("--enforce-eager")
    with pytest.raises(ValueError, match="compiled BF16"):
        validate_resident(server, server, native)
    server["command"].pop()
    native["quantization"] = "fp4"
    with pytest.raises(ValueError, match="native audit"):
        validate_resident(server, server, native)
