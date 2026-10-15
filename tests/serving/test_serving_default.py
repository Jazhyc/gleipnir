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


def last_query_fixture(root):
    source, report, _ = fixture(root)
    command = json.loads(report.read_text())
    additional = command["command"][-1]
    command["command"][-2:] = [
        "--worker-cls",
        "experiments.precision.AttentionPrecisionWorker",
        "--additional-config",
        additional,
    ]
    report.write_text(json.dumps(command))
    trim_source = root / "src/last_query.py"
    trim_source.write_text("final layer\n")
    receipt = root / "results/trim/last_query.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text('{"passed": true}\n')
    selection_path = root / "experiments/b200_inference_benchmark/serving_default.json"
    selection = json.loads(selection_path.read_text())

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    selection["artifact_bindings"] = {
        "results/fp8/summary.json": digest(report),
        "results/trim/last_query.json": digest(receipt),
    }
    selection["last_query_attention"] = {
        "worker": "experiments.trim.LastQueryPrecisionWorker",
        "replaces_worker": "experiments.precision.AttentionPrecisionWorker",
        "validation": "results/trim/last_query.json",
        "sources": {"src/last_query.py": digest(trim_source)},
    }
    selection_path.write_text(json.dumps(selection))
    return trim_source, selection_path


def test_default_applies_bound_last_query_attention(tmp_path):
    last_query_fixture(tmp_path)
    _, command = selected_serving_default(tmp_path)
    assert command[command.index("--worker-cls") + 1] == (
        "experiments.trim.LastQueryPrecisionWorker"
    )
    additional = json.loads(command[-1])
    assert additional["last_query_attention"]["validation"] == (
        "results/trim/last_query.json"
    )
    assert "src/last_query.py" in additional["gleipnir_frost_fp4"]
    assert "src/kernel.py" in additional["gleipnir_frost_fp4"]


def test_default_rejects_last_query_source_drift(tmp_path):
    trim_source, _ = last_query_fixture(tmp_path)
    trim_source.write_text("changed\n")
    with pytest.raises(ValueError, match="last-query source drift"):
        selected_serving_default(tmp_path)


def test_default_rejects_unexpected_replaced_worker(tmp_path):
    _, selection_path = last_query_fixture(tmp_path)
    selection = json.loads(selection_path.read_text())
    selection["last_query_attention"]["replaces_worker"] = "other.Worker"
    selection_path.write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="unexpected worker"):
        selected_serving_default(tmp_path)


def test_default_stacks_gdn_route_on_last_query_worker(tmp_path):
    _, selection_path = last_query_fixture(tmp_path)
    route_source = tmp_path / "src/gdn_route.py"
    route_source.write_text("gate\n")
    evidence = tmp_path / "results/route/summary.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("{}\n")
    selection = json.loads(selection_path.read_text())

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    selection["artifact_bindings"]["results/route/summary.json"] = digest(evidence)
    selection["gdn_route"] = {
        "cp_min_tokens": 8192,
        "worker": "experiments.route.GatedPrecisionWorker",
        "replaces_worker": "experiments.trim.LastQueryPrecisionWorker",
        "evidence": ["results/route/summary.json"],
        "sources": {"src/gdn_route.py": digest(route_source)},
    }
    selection_path.write_text(json.dumps(selection))
    _, command = selected_serving_default(tmp_path)
    assert command[command.index("--worker-cls") + 1] == (
        "experiments.route.GatedPrecisionWorker"
    )
    additional = json.loads(command[-1])
    assert additional["gdn_route"] == {"cp_min_tokens": 8192}
    assert "last_query_attention" in additional
    _, control = selected_serving_default(tmp_path, last_query=False)
    assert "gdn_route" not in json.loads(control[-1])
    route_source.write_text("changed\n")
    with pytest.raises(ValueError, match="worker layer source drift"):
        selected_serving_default(tmp_path)
