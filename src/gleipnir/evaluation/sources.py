"""Source fingerprints for moved evaluators, including their legacy aliases.

New execution contracts must bind implementations and score helpers as well as
compatibility modules. Historical manifests and drift checks remain unchanged.
"""

_SOURCES = "src/gleipnir/evaluation/sources.py"
_PACKAGE = "src/gleipnir/evaluation/__init__.py"
_PROBABILITIES = "src/gleipnir/evaluation/probabilities.py"
_CALIBRATION = "src/gleipnir/evaluation/calibration.py"

DECISION_SOURCE_PATHS = (
    "src/gleipnir/decision_surface.py",
    "src/gleipnir/evaluation/decision_surface.py",
    _SOURCES,
    _PACKAGE,
)

BINARY_SOURCE_PATHS = (
    "src/gleipnir/binary_evaluation.py",
    "src/gleipnir/evaluation/binary.py",
    "src/gleipnir/evaluation/metrics.py",
    _PROBABILITIES,
    _SOURCES,
    _PACKAGE,
)

SCORING_SOURCE_PATHS = (
    "src/gleipnir/monitoring_scoring.py",
    "src/gleipnir/evaluation/scoring.py",
    _PROBABILITIES,
    *DECISION_SOURCE_PATHS,
)

PREFERENCE_SOURCE_PATHS = (
    "src/gleipnir/judge_injection_metrics.py",
    "src/gleipnir/evaluation/preferences.py",
    _CALIBRATION,
    _SOURCES,
    _PACKAGE,
)
