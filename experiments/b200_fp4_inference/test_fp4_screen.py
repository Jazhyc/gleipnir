"""FP4 backend/source identity and per-kernel ranking comparisons."""

import json

import pytest

from experiments.b200_fp4_inference.run import EXPERIMENT, resolve_condition
from gleipnir.inference_benchmark import ranking_comparison


def test_fp4_requires_cudnn_and_binds_compiler_identity():
    condition = json.loads((EXPERIMENT / "config.json").read_text())
    a = resolve_condition(condition, {"wrapper": "first"})
    b = resolve_condition(condition, {"wrapper": "second"})
    assert a["extra_server_args"] != b["extra_server_args"]
    assert a["allow_finite_parity_diagnostic"]
    with pytest.raises(ValueError, match="requires native"):
        resolve_condition({**condition, "linear_backend": "emulation"}, {})


def test_auroc_catches_reordering_and_preserves_undefined_sources():
    rows = [
        {
            "id": str(i),
            "prompt_sha256": str(i),
            "dataset": "single" if i == 4 else "both",
            "label": y,
        }
        for i, y in enumerate([0, 0, 1, 1, 1])
    ]

    def outputs(scores):
        return [
            {
                "id": r["id"],
                "prompt_sha256": r["prompt_sha256"],
                "score": v,
                "margin": v,
            }
            for r, v in zip(rows, scores, strict=True)
        ]

    baseline = outputs([0.1, 0.2, 0.8, 0.9, 0.7])
    candidate = outputs([0.1, 0.8, 0.2, 0.9, 0.7])
    result = ranking_comparison(rows, [baseline], [candidate])
    assert result["auroc_delta"]["macro"] == pytest.approx(-0.25)
    assert result["undefined_auroc_sources"] == ["single"]
    assert result["auroc_delta"]["per_source"]["single"] is None
    with pytest.raises(ValueError, match="identity drift"):
        ranking_comparison(list(reversed(rows)), [baseline], [candidate])
