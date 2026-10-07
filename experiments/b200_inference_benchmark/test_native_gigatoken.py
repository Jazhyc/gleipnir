"""Native frontend keeps truncation and rejects unsupported token insertion."""

import json

import numpy as np
import pytest

from gleipnir.serving_gigatoken import NativeEncoder, configure_frontend
from gleipnir.serving_runtime import sha


def encoder(side="right", processor=None):
    class Backend:
        def to_str(self):
            return json.dumps({"post_processor": processor})

    class Tokenizer:
        truncation_side = side
        backend_tokenizer = Backend()

    class Native:
        def __init__(self, tokenizer):
            pass

        def encode(self, text):
            return np.array([1, 2, 3, 4], dtype=np.uint32)

    return NativeEncoder(Tokenizer(), factory=Native)


@pytest.mark.parametrize("side,expected", [("left", [3, 4]), ("right", [1, 2])])
def test_native_encoder_preserves_truncation_side_and_zero_length(side, expected):
    value = encoder(side, {"type": "ByteLevel"})
    assert value.encode("text", truncation=True, max_length=2) == expected
    assert value.encode("text", truncation=True, max_length=0) == []
    assert value.encode("text", add_special_tokens=True) == [1, 2, 3, 4]
    assert value.calls == 3


def test_native_encoder_rejects_postprocessors_that_insert_tokens():
    with pytest.raises(ValueError, match="token-neutral"):
        encoder(processor={"type": "TemplateProcessing"})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"truncation": True},
        {"max_length": -1},
        {"max_length": True},
        {"truncation": "longest_first"},
        {"add_special_tokens": "yes"},
    ],
)
def test_native_encoder_rejects_unsupported_flags(kwargs):
    with pytest.raises(ValueError):
        encoder().encode("text", **kwargs)


def test_frontend_binds_package_and_changes_only_cpu_launch(tmp_path):
    source = tmp_path / "src/gleipnir/serving_gigatoken.py"
    source.parent.mkdir(parents=True)
    source.write_text("validated implementation")
    entry = tmp_path / "experiments/b200_inference_benchmark/frontend_server.py"
    entry.parent.mkdir(parents=True)
    entry.write_text("CPU wrapper")
    package = tmp_path / "vendor"
    module = package / "gigatoken/backend.so"
    module.parent.mkdir(parents=True)
    module.write_text("pinned native extension")
    validation = tmp_path / "validation.json"
    validation.write_text(
        json.dumps(
            {
                "exact": True,
                "version": "0.10.0",
                "source_sha256": sha(source),
                "package_files": {"gigatoken/backend.so": sha(module)},
            }
        )
    )
    front = {
        "backend": "gigatoken_native",
        "validation": "validation.json",
        "package_path": str(package),
        "receipt_path": "frontend.json",
    }
    command = [
        "python",
        "-m",
        "experiments.b200_attention_gdn_serving.server",
        "--additional-config",
        '{"unchanged_gpu":"bound"}',
    ]
    env = {"PYTHONPATH": "src"}
    receipt = configure_frontend(tmp_path, front, command, env)
    assert receipt["version"] == "0.10.0"
    assert command[-1] == '{"unchanged_gpu":"bound"}'
    assert command[2] == "experiments.b200_inference_benchmark.frontend_server"
    assert env["PYTHONPATH"] == f"{package}:src"
    module.write_text("changed extension")
    with pytest.raises(ValueError, match="package drift"):
        configure_frontend(tmp_path, front, command, env)
