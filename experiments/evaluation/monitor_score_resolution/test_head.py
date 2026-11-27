"""Prove projection precedes BF16 output rounding and contract inputs stay fixed."""

import json
from pathlib import Path

import pytest
import torch

from gleipnir.serving.bf16_monitor import command as baseline_command

from .head import project
from .run import install_head_override


def test_fp32_head_recovers_differences_lost_by_bf16_output():
    hidden = torch.tensor([[1.0, 1.0], [1.0, 1.0078125]], dtype=torch.bfloat16)
    original_rows = torch.tensor([[16.0, 1.0], [16.0, 1.0078125]], dtype=torch.bfloat16)
    output = project(hidden, original_rows.float())
    assert output.dtype == torch.float32
    assert torch.equal(output, hidden.float() @ original_rows.float().T)
    assert output[0, 1] != output[1, 1]
    assert torch.unique(output.to(torch.bfloat16)).numel() == 1
    assert torch.equal(original_rows, original_rows.float().to(torch.bfloat16))


@pytest.mark.parametrize(
    "hidden_dtype,weight_dtype",
    [(torch.float32, torch.float32), (torch.bfloat16, torch.bfloat16)],
)
def test_reject_unexpected_precision(hidden_dtype, weight_dtype):
    with pytest.raises(ValueError, match="expected BF16"):
        project(
            torch.ones(1, 2, dtype=hidden_dtype), torch.ones(2, 2, dtype=weight_dtype)
        )


def test_command_changes_only_worker_and_head_precision(monkeypatch):
    from gleipnir.serving import bf16_monitor

    parent = [
        "python",
        "-m",
        "old.server",
        "--worker-cls",
        "OldWorker",
        "--model",
        "/tmp/strict",
        "--additional-config",
        json.dumps(
            {"monitor_score": {}, "runtime_migration": {}, "serving_condition": {}}
        ),
    ]
    expected = baseline_command(parent, {})
    monkeypatch.setattr(bf16_monitor, "command", baseline_command)
    install_head_override()
    candidate = bf16_monitor.command(parent, {})
    i = candidate.index("--additional-config") + 1
    additional = json.loads(candidate[i])
    assert additional.pop("monitor_head_precision") == "fp32"
    assert additional == json.loads(expected[i])
    candidate[i] = expected[i]
    j = candidate.index("--worker-cls") + 1
    assert candidate[j].endswith("Fp32HeadWorker")
    candidate[j] = expected[j]
    assert candidate == expected


def test_readonly_adapter_and_id_ood_membership():
    from .run import HeadContext

    ctx = HeadContext(
        Path("/workspace/gleipnir"),
        Path("config.yaml"),
        {"parent_campaign": "results/strict", "campaign_id": "test"},
    )
    assert ctx.adapter == Path("/workspace/gleipnir/results/strict/4b/monitor")
    assert ctx.splits == ("id", "ood")
