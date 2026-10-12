import math

import pytest

from .metrics import row_fit


def test_matching_teacher_has_zero_kl():
    for q in (0.01, 0.5, 0.99):
        r = row_fit(q, math.log(q / (1 - q)))
        assert r["kl"] == pytest.approx(0, abs=1e-12)
        assert r["ce"] == pytest.approx(r["teacher_entropy"])


def test_extreme_margins_do_not_overflow_or_clip():
    assert row_fit(1, -1000)["kl"] == 1000
    assert row_fit(0, 1000)["kl"] == 1000
    assert row_fit(1, 1000)["kl"] == 0
    assert row_fit(0.5, 0)["ce"] == pytest.approx(math.log(2))


@pytest.mark.parametrize("width", [2560, 4096])
def test_fp32_projection_supports_both_pinned_model_widths(width):
    import torch

    from experiments.monitor_score_resolution.head import project

    hidden = torch.ones(1, width, dtype=torch.bfloat16)
    weights = torch.ones(2, width, dtype=torch.float32)
    scores = project(hidden, weights)
    assert scores.dtype == torch.float32
    assert scores.shape == (1, 2)
    assert torch.equal(scores, torch.full((1, 2), float(width)))


def test_fit_context_uses_separate_receipt_and_original_weights():
    from pathlib import Path

    from .run import FitContext

    ctx = FitContext(
        Path("/workspace/gleipnir"),
        Path("config.yaml"),
        {"campaign_id": "final-training-fit02-9b", "size": "9b"},
    )
    assert ctx.adapter == ctx.output / "adapter"
    assert str(ctx.source_adapter).endswith("student_injection_awareness/9b/regular")
    assert ctx.adapter != ctx.source_adapter
