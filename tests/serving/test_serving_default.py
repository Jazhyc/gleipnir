"""The current serving selection never rewrites frozen comparison controls."""

import hashlib
import json

import pytest

from gleipnir.serving.reference import selected_serving_default


def fixture(root):
    source = root / "src/kernel.py"
    source.parent.mkdir(parents=True)
    source.write_text("validated arithmetic\n")
    report = root / "results/fp8/summary.json"
    report.parent.mkdir(parents=True)
    report.write_text(
        json.dumps(
            {
                "status": "complete",
                "precision": "fp8",
                "command": [
                    "python",
                    "--max-model-len",
                    "32768",
                    "--runner",
                    "pooling",
                    "--additional-config",
                    json.dumps(
                        {
                            "serving_condition": {
                                "attention_projection_precision": "fp8",
                                "gdn_projection_precision": "fp4",
                                "attention_precision": "mxfp8",
                            },
                            "gleipnir_frost_fp4": {
                                "src/kernel.py": hashlib.sha256(
                                    source.read_bytes()
                                ).hexdigest()
                            },
                        }
                    ),
                ],
            }
        )
    )
    selection = root / "experiments/b200_inference_benchmark/serving_default.json"
    selection.parent.mkdir(parents=True)
    selection.write_text(
        json.dumps(
            {
                "quality_status": "user_accepted_finite",
                "recipe_summary": "results/fp8/summary.json",
                "artifact_bindings": {
                    "results/fp8/summary.json": hashlib.sha256(
                        report.read_bytes()
                    ).hexdigest()
                },
            }
        )
    )
    baseline = selection.parent / "baseline.json"
    baseline.write_text('{"frozen":"original FP4 comparison"}\n')
    return source, report, baseline


def test_default_resolves_fp8_without_touching_frozen_baseline(tmp_path):
    _, _, baseline = fixture(tmp_path)
    before = baseline.read_bytes()
    selection, command = selected_serving_default(tmp_path)
    assert selection["quality_status"] == "user_accepted_finite"
    assert (
        json.loads(command[-1])["serving_condition"]["attention_projection_precision"]
        == "fp8"
    )
    assert baseline.read_bytes() == before


def test_default_rejects_arithmetic_source_drift(tmp_path):
    source, _, _ = fixture(tmp_path)
    source.write_text("changed arithmetic\n")
    with pytest.raises(ValueError, match="runtime source drift"):
        selected_serving_default(tmp_path)


def test_default_rejects_reference_artifact_drift(tmp_path):
    _, report, _ = fixture(tmp_path)
    report.write_text("{}")
    with pytest.raises(ValueError, match="artifact drift"):
        selected_serving_default(tmp_path)
