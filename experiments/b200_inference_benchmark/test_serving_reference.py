"""Reference defaults must stay source-bound and preserve resident metadata."""

import hashlib
import json

import pytest

from experiments.b200_inference_benchmark.run import compatible_server_command
from gleipnir.serving_reference import selected_host_components


def test_selected_components_bind_receipts_and_keep_resident_path(tmp_path):
    baseline = tmp_path / "experiments/b200_inference_benchmark/baseline.json"
    baseline.parent.mkdir(parents=True)
    receipt = tmp_path / "results/validation.json"
    receipt.parent.mkdir()
    receipt.write_text('{"passed":true}')
    validation = {
        "validation": "results/validation.json",
        "validation_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
    }
    baseline.write_text(
        json.dumps(
            {
                "frontend": {"backend": "gigatoken_native", **validation},
                "host_wrapper": validation,
            }
        )
    )
    frontend, host = selected_host_components(tmp_path, tmp_path / "out")
    assert frontend["receipt_path"] == str(tmp_path / "out/frontend.json")
    assert host == {"validation": "results/validation.json"}
    assert (
        selected_host_components(
            tmp_path,
            tmp_path / "out",
            resident={"frontend": {"receipt_path": "resident.json"}},
        )[0]["receipt_path"]
        == "resident.json"
    )
    receipt.write_text('{"passed":false}')
    with pytest.raises(ValueError, match="checksum drift"):
        selected_host_components(tmp_path, tmp_path / "out")


def test_legacy_gpu_reference_has_no_new_cpu_defaults(tmp_path):
    path = tmp_path / "experiments/b200_inference_benchmark/baseline.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    assert selected_host_components(tmp_path, tmp_path / "out") == (None, None)


def test_resident_command_accepts_control_changes_but_rejects_arithmetic():
    def command(name, source="same", precision="fp4"):
        return [
            "python",
            "-m",
            "server",
            "--additional-config",
            json.dumps(
                {
                    "serving_condition": {
                        "name": name,
                        "baseline": name,
                        "attention_precision": precision,
                        "profiler_config": {"with_stack": False},
                    },
                    "gleipnir_frost_fp4": {
                        "src/kernel.py": source,
                        "experiment/run.py": name,
                    },
                }
            ),
        ]

    assert compatible_server_command(command("old"), command("new"))
    assert not compatible_server_command(
        command("old"), command("new", source="changed")
    )
    assert not compatible_server_command(
        command("old"), command("new", precision="fp8")
    )
