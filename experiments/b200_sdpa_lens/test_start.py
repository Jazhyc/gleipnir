"""A Lens restore cannot silently switch the SDPA model or BF16 arithmetic."""

import json
from copy import deepcopy

import pytest

from experiments.b200_sdpa_lens.start import lens_command


def parent() -> dict:
    condition = {
        "quantization": None,
        "attention_precision": "bf16",
        "attention_projection_precision": "bf16",
        "gdn_projection_precision": "bf16",
        "mlp_precision": "bf16",
    }
    return {
        "serving_precision": "bf16",
        "command": [
            "old-python",
            "-m",
            "gleipnir.serving.bf16_server",
            "--model",
            "/tmp/sdpa",
            "--worker-cls",
            "gleipnir.serving.bf16_worker.Bf16Worker",
            "--additional-config",
            json.dumps({"serving_condition": condition}),
            "--no-enable-prefix-caching",
            "--max-num-seqs",
            "128",
        ],
    }


def test_lens_restore_preserves_parent_command():
    original = parent()
    before = deepcopy(original)
    command = lens_command(original, "/tmp/sdpa")
    assert original == before
    assert command[2] == "experiments.b200_sdpa_lens.server"
    assert command[3 : len(before["command"])] == before["command"][3:]
    assert command[-3:] == [
        "--enforce-eager",
        "--worker-extension-cls",
        "gleipnir.serving.lens_worker.MonitorLensExtension",
    ]


@pytest.mark.parametrize(
    "field",
    [
        "quantization",
        "attention_precision",
        "attention_projection_precision",
        "gdn_projection_precision",
        "mlp_precision",
    ],
)
def test_quantized_parent_is_rejected(field):
    original = parent()
    condition = json.loads(
        original["command"][original["command"].index("--additional-config") + 1]
    )
    condition["serving_condition"][field] = "fp4"
    original["command"][original["command"].index("--additional-config") + 1] = (
        json.dumps(condition)
    )
    with pytest.raises(ValueError, match="unquantized BF16"):
        lens_command(original, "/tmp/sdpa")


def test_other_adapter_is_rejected():
    with pytest.raises(ValueError, match="matching unquantized"):
        lens_command(parent(), "/tmp/other")
