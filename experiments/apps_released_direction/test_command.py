import json

import pytest

from experiments.apps_released_direction.run import lens_command


def template(worker: str = "gleipnir.serving.bf16_worker.Bf16Worker") -> dict:
    extra = {"serving_condition": {"merged_model": "/tmp/old", "quantization": None}}
    return {
        "serving_precision": "bf16",
        "command": [
            "python",
            "-m",
            "gleipnir.serving.bf16_server",
            "--model",
            "/tmp/old",
            "--worker-cls",
            worker,
            "--additional-config",
            json.dumps(extra),
        ],
    }


def test_lens_command_swaps_model_and_adds_lens() -> None:
    command = lens_command(template(), "/tmp/new")
    assert command[2] == "experiments.b200_sdpa_lens.server"
    assert command[command.index("--model") + 1] == "/tmp/new"
    extra = json.loads(command[command.index("--additional-config") + 1])
    assert extra["serving_condition"]["merged_model"] == "/tmp/new"
    assert "/tmp/old" not in " ".join(command)
    assert command[-3:] == [
        "--enforce-eager",
        "--worker-extension-cls",
        "gleipnir.serving.lens_worker.MonitorLensExtension",
    ]


def test_lens_command_requires_bf16_worker() -> None:
    with pytest.raises(ValueError):
        lens_command(template("other.Worker"), "/tmp/new")
