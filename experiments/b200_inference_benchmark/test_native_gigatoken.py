"""Native frontend keeps truncation and rejects unsupported token insertion."""

import json

import numpy as np
import pytest

from gleipnir.serving_gigatoken import (
    FrontendControl,
    NativeEncoder,
    configure_frontend,
)
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


def test_control_routes_each_backend_and_records_completed_encodes():
    class Encoder:
        def __init__(self, value):
            self.value = value

        def encode(self, text, **kwargs):
            assert text == "text" and kwargs == {"add_special_tokens": False}
            return [self.value]

    control = FrontendControl()
    native, hf = Encoder(1), Encoder(2)
    assert control.encode(native, hf, "text", add_special_tokens=False) == [1]
    assert control.toggle()["mode"] == "hf"
    assert control.encode(native, hf, "text", add_special_tokens=False) == [2]
    state = control.toggle()
    assert state["mode"] == "native" and state["generation"] == 2
    assert state["calls"] == {"native": 1, "hf": 1}
    assert state["active_encodes"] == 0


def test_control_rejects_switch_during_encoding_and_drains_after_failure():
    control = FrontendControl()

    class Encoder:
        def encode(self, text, **kwargs):
            with pytest.raises(RuntimeError, match="drain active"):
                control.toggle()
            raise ValueError("encoding failure")

    with pytest.raises(ValueError, match="encoding failure"):
        control.encode(Encoder(), None, "text")
    assert control.snapshot()["active_encodes"] == 0
    assert control.toggle()["mode"] == "hf"
