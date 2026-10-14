"""CPU checks for the 1M command, envelope and node-environment receipts."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.b200_context_million import envelope
from experiments.b200_context_million.environment import (
    drift,
    parse_gpus,
    summarize_samples,
)

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = json.loads((Path(__file__).with_name("config.json")).read_text())
PHASES = {p["name"]: p for p in SETTINGS["phases"]}


def selected_command() -> list[str]:
    summary = ROOT / "results/b200_attention_precision/precision01/fp8/summary.json"
    if not summary.exists():
        pytest.skip("archived FP8 default receipt is not available")
    return json.loads(summary.read_text())["command"]


def test_native_phase_changes_only_length_and_worker():
    from experiments.b200_context_million.run import WORKER, phase_command

    parent = selected_command()
    command = phase_command(parent, PHASES["native"], 32768)
    changed = [
        i for i, (a, b) in enumerate(zip(parent, command, strict=True)) if a != b
    ]
    names = {command[i - 1] for i in changed if i}
    assert names == {"--max-model-len", "--worker-cls", "--additional-config"}
    assert command[command.index("--max-model-len") + 1] == "262144"
    assert command[command.index("--worker-cls") + 1] == WORKER
    overrides = json.loads(command[command.index("--hf-overrides") + 1])
    assert overrides["text_config"]["rope_parameters"]["rope_type"] == "default"


def test_yarn_phase_scales_rope_and_keeps_mrope():
    from experiments.b200_context_million.run import phase_command

    command = phase_command(selected_command(), PHASES["yarn"], 32768)
    assert command[command.index("--max-model-len") + 1] == "1048576"
    assert command[command.index("--max-num-batched-tokens") + 1] == "32768"
    text = json.loads(command[command.index("--hf-overrides") + 1])["text_config"]
    assert text["max_position_embeddings"] == 1048576
    rope = text["rope_parameters"]
    assert rope["rope_type"] == "yarn"
    assert rope["factor"] * rope["original_max_position_embeddings"] == 1048576
    assert rope["mrope_section"] == [11, 11, 10] and rope["mrope_interleaved"]


def test_phase_rejects_uncovered_limits():
    from experiments.b200_context_million.run import phase_command

    parent = selected_command()
    with pytest.raises(ValueError, match="position limit"):
        phase_command(parent, {**PHASES["native"], "context_limit": 524288}, 32768)
    too_long = {**PHASES["yarn"], "context_limit": 2 * 1048576}
    with pytest.raises(ValueError, match="YaRN"):
        phase_command(parent, too_long, 32768)


def test_envelope_requires_complete_history_coverage(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("x = 1\n")
    import hashlib

    checks = [
        {
            "query_tokens": q,
            "history_tokens": n,
            "finite": True,
            "quantized_relative_l2": 0.002,
        }
        for q, n in [envelope.IRREGULAR, *[(32768, h) for h in envelope.HISTORIES]]
    ]
    receipt = {
        "passed": True,
        "context_limit": envelope.LIMIT,
        "short_bitwise_cases": 2,
        "checks": checks,
        "sources": {str(source): hashlib.sha256(b"x = 1\n").hexdigest()},
    }
    envelope.validate(receipt, tmp_path, 1048576)
    with pytest.raises(ValueError, match="coverage"):
        envelope.validate({**receipt, "checks": checks[:-1]}, tmp_path, 1048576)
    source.write_text("x = 2\n")
    with pytest.raises(ValueError, match="source changed"):
        envelope.validate(receipt, tmp_path, 1048576)


def test_enable_changes_only_length_bound():
    cleared = []
    module = SimpleNamespace(
        MAX_BATCH=128,
        MAX_LENGTH=32768,
        forward_plan=SimpleNamespace(cache_clear=lambda: cleared.append(True)),
    )
    envelope.enable(module, 262144)
    assert module.MAX_LENGTH == 262144 and module.MAX_BATCH == 128 and cleared
    with pytest.raises(ValueError):
        envelope.enable(module, 2 * envelope.LIMIT)


def test_gpu_identity_and_drift():
    xml = """<nvidia_smi_log><driver_version>580.126.20</driver_version>
    <cuda_version>13.0</cuda_version><gpu id="0">
    <product_name>NVIDIA B200</product_name>
    <uuid>GPU-x</uuid><vbios_version>97.00</vbios_version>
    <max_clocks><sm_clock>1965 MHz</sm_clock></max_clocks></gpu></nvidia_smi_log>"""
    gpus = parse_gpus(xml)
    assert gpus[0]["driver_version"] == "580.126.20"
    assert gpus[0]["clock_sm_max"] == "1965 MHz" and gpus[0]["serial"] is None
    other = [{**gpus[0], "driver_version": "595.91.07"}]
    changes = drift({"gpus": gpus}, {"gpus": other})
    assert set(changes) == {"driver_version"}


def test_sample_summary_reports_throttling():
    text = (
        "timestamp, clocks.sm [MHz], clocks.mem [MHz], power.draw [W], "
        "temperature.gpu, utilization.gpu [%], memory.used [MiB], "
        "clocks_event_reasons.active\n"
        "t, 1965, 3996, 700.1, 60, 100, 1000, 0x0000000000000000\n"
        "t, 1800, 3996, 990.5, 71, 100, 1000, 0x0000000000000004\n"
    )
    summary = summarize_samples(text)
    assert summary["samples"] == 2 and summary["sm_clock_min_mhz"] == 1800
    assert summary["power_max_w"] == 990.5
    assert summary["event_reasons"] == ["0x0000000000000000", "0x0000000000000004"]
