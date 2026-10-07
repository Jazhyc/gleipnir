"""Graph launch preserves score/mutation contracts and binds saved controls."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from experiments.b200_score_graphs.run import graph_command
from gleipnir.serving.reference import selected_score_reference


def test_graph_command_retains_endpoint_precision_and_parent():
    settings = json.loads((Path(__file__).parent / "config.json").read_text())
    parent = [
        "python",
        "-m",
        "experiments.b200_mutation_analysis.server",
        "--runner",
        "pooling",
        "--worker-cls",
        "score",
        "--additional-config",
        json.dumps(
            {
                "qk_mutation_analysis": True,
                "monitor_score": {"token_ids": [15, 16]},
                "gleipnir_frost_fp4": {},
                "serving_condition": {"attention_precision": "mxfp8"},
            }
        ),
    ]
    saved = copy.deepcopy(parent)
    command = graph_command(
        parent, settings, {"experiments/b200_score_graphs/worker.py": "worker-sha"}
    )
    assert parent == saved
    extra = json.loads(command[command.index("--additional-config") + 1])
    assert extra["qk_mutation_analysis"] is True
    assert extra["monitor_score"] == {"token_ids": [15, 16]}
    assert extra["serving_condition"]["attention_precision"] == "mxfp8"
    assert json.loads(command[-1])["cudagraph_capture_sizes"][-1] == 32768
    with pytest.raises(ValueError, match="already overrides"):
        graph_command(command, settings, {})
    parent[2] = "experiments.b200_monitor_score.server"
    with pytest.raises(ValueError, match="repaired"):
        graph_command(parent, settings, {})


def test_score_reference_rejects_prediction_and_workload_drift(tmp_path):
    directory = tmp_path / "results/control"
    directory.mkdir(parents=True)
    (directory / "summary.json").write_text(
        json.dumps({"status": "complete", "score_audit_passed": True})
    )
    for c, count in ((1, 3), (128, 6)):
        for i in range(count):
            (directory / f"c{c}_repeat{i}.json").write_text("[]")
    manifest = tmp_path / "data/b200_inference_benchmark/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")

    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    selection = {
        "endpoint": "/v1/monitor/score",
        "results": "results/control",
        "manifest_sha256": sha(manifest),
        "artifact_bindings": {
            str(p.relative_to(tmp_path)): sha(p) for p in directory.iterdir()
        },
    }
    baseline = tmp_path / "experiments/b200_inference_benchmark/baseline.json"
    baseline.parent.mkdir(parents=True)
    baseline.write_text(json.dumps(selection))
    assert selected_score_reference(tmp_path) == selection
    (directory / "c128_repeat5.json").write_text("[0]")
    with pytest.raises(ValueError, match="artifact drift"):
        selected_score_reference(tmp_path)
    (directory / "c128_repeat5.json").write_text("[]")
    manifest.write_text("changed")
    with pytest.raises(ValueError, match="workload drift"):
        selected_score_reference(tmp_path)
