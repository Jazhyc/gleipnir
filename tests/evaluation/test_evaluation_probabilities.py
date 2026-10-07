"""Freeze score numerics and explicit logprob requirements across the refactor."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from gleipnir.evaluation.probabilities import (
    normalized_binary_probability,
    score_from_output,
)
from gleipnir.evaluation.sources import (
    BINARY_SOURCE_PATHS,
    DECISION_SOURCE_PATHS,
    PREFERENCE_SOURCE_PATHS,
    SCORING_SOURCE_PATHS,
)
from tests.helpers.paths import ROOT

# Captured from score_from_output before the package move. Keep exact FP64
# outputs: near-half scores and saturated tails affect ties and cache validation.
SCORE_RECEIPTS = [
    (0, 0, "0x1.0000000000000p-1"),
    (-2, -3, "0x1.136561454ba86p-2"),
    (-3, -2, "0x1.764d4f5d5a2bdp-1"),
    (-10000, -10002, "0x1.e84152bac31aep-4"),
    (-10000, -9998, "0x1.c2f7d5a8a79c9p-1"),
    (0, -80, "0x1.7fd974d372e45p-116"),
    (-80, 0, "0x1.0000000000000p+0"),
    (0, -81, "0x1.7fd974d372e45p-116"),
    (-81, 0, "0x1.0000000000000p+0"),
    (-0.01, -4, "0x1.29980abc45e5fp-6"),
    (-1.3862943611198906, -0.2876820724517809, "0x1.7ffffffffffffp-1"),
]


@pytest.mark.parametrize("zero,one,expected", SCORE_RECEIPTS)
def test_score_conversion_matches_frozen_float64_outputs(zero, one, expected):
    output = SimpleNamespace(outputs=[SimpleNamespace(logprobs=[{48: zero, 49: one}])])
    assert normalized_binary_probability(zero, one).hex() == expected
    assert score_from_output(output, [48, 49]).hex() == expected


@pytest.mark.parametrize("representation", [float, dict, SimpleNamespace])
def test_score_accepts_supported_logprob_formats(representation):
    def wrap(value):
        return value if representation is float else representation(logprob=value)

    output = SimpleNamespace(
        outputs=[SimpleNamespace(logprobs=[{"48": wrap(-2), "49": wrap(-3)}])]
    )
    assert score_from_output(output, [48, 49]).hex() == "0x1.136561454ba86p-2"


@pytest.mark.parametrize(
    "outputs,message",
    [
        ([], "no first-token logprobs"),
        ([SimpleNamespace(logprobs=None)], "no first-token logprobs"),
        ([SimpleNamespace(logprobs=[{}])], "requested token logprobs"),
        ([SimpleNamespace(logprobs=[{48: -2}])], "requested token logprobs"),
    ],
)
def test_score_rejects_missing_explicit_logprob_evidence(outputs, message):
    with pytest.raises(RuntimeError, match=message):
        score_from_output(SimpleNamespace(outputs=outputs), [48, 49])


def test_probability_helpers_do_not_import_model_or_metric_stacks():
    code = """
import sys
import gleipnir.evaluation.probabilities
assert not {'torch', 'vllm', 'pandas', 'sklearn', 'numpy'} & set(sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)


@pytest.mark.parametrize(
    "sources,required",
    [
        (BINARY_SOURCE_PATHS, {"binary", "metrics", "probabilities"}),
        (SCORING_SOURCE_PATHS, {"scoring", "decision_surface", "probabilities"}),
        (PREFERENCE_SOURCE_PATHS, {"preferences", "calibration"}),
        (DECISION_SOURCE_PATHS, {"decision_surface"}),
    ],
)
def test_source_fingerprints_include_implementations_and_helpers(sources, required):
    root = ROOT
    assert all((root / path).is_file() for path in sources)
    canonical = {Path(p).stem for p in sources if "/evaluation/" in p}
    assert required | {"sources"} <= canonical
