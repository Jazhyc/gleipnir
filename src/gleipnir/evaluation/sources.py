"""Source fingerprints for moved evaluators and the central alias registry.

New execution contracts must bind implementations and score helpers as well as
alias registration. Historical manifests and drift checks remain unchanged.
"""

_COMMON = (
    "src/gleipnir/__init__.py",
    "src/gleipnir/_compat.py",
    "src/gleipnir/evaluation/__init__.py",
    "src/gleipnir/evaluation/sources.py",
)
_PROBABILITIES = "src/gleipnir/evaluation/probabilities.py"
_CALIBRATION = "src/gleipnir/evaluation/calibration.py"

DECISION_SOURCE_PATHS = (
    "src/gleipnir/evaluation/decision_surface.py",
    *_COMMON,
)

BINARY_SOURCE_PATHS = (
    "src/gleipnir/evaluation/binary.py",
    "src/gleipnir/evaluation/metrics.py",
    _PROBABILITIES,
    *_COMMON,
)

SCORING_SOURCE_PATHS = (
    "src/gleipnir/evaluation/scoring.py",
    _PROBABILITIES,
    *DECISION_SOURCE_PATHS,
)

PREFERENCE_SOURCE_PATHS = (
    "src/gleipnir/evaluation/preferences.py",
    _CALIBRATION,
    *_COMMON,
)
