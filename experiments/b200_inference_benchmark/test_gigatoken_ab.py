"""Paired analysis and control receipts must preserve GPU process identity."""

import json

import pytest

from experiments.b200_inference_benchmark import gigatoken_ab


def test_ratios_compare_corresponding_pairs_in_the_correct_direction():
    result = gigatoken_ab.paired_ratios([100.0, 200.0], [200.0, 200.0])
    assert result["ratios"] == [2.0, 1.0]
    assert result["median"] == 1.5
    assert result["bootstrap_95_interval"] == [1.0, 2.0]
    with pytest.raises(ValueError):
        gigatoken_ab.paired_ratios([100.0], [100.0, 200.0])


def test_controller_rejects_a_changed_gpu_engine_and_non_control_receipt(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(gigatoken_ab, "ROOT", tmp_path)
    monkeypatch.setattr(gigatoken_ab.os, "kill", lambda pid, sig: None)
    root = tmp_path / "results/b200_attention_gdn_serving"
    root.mkdir(parents=True)
    receipt = tmp_path / "frontend.json"
    server = {"pid": 100, "frontend": {"receipt_path": str(receipt)}}
    (root / "server.json").write_text(json.dumps(server))
    (root / "loaded_precision.json").write_text('{"worker_pid":200}')
    state = {"pid": 100, "ab_control": {"mode": "native", "generation": 0}}
    receipt.write_text(json.dumps(state))
    controller = gigatoken_ab.ModeController(server, 200)
    assert controller.read()["mode"] == "native"
    (root / "loaded_precision.json").write_text('{"worker_pid":201}')
    with pytest.raises(ValueError, match="GPU engine changed"):
        controller.read()
    receipt.write_text('{"pid":100}')
    with pytest.raises(ValueError, match="identity or control drift"):
        controller.read()
