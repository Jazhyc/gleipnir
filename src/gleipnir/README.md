# Shared source map

Place new shared code in the package matching its responsibility. Historical
import names resolve lazily through the single [_compat.py](_compat.py) registry.
Use canonical package paths for new code; moved implementations have one home.

| Package | Responsibilities | Starting points |
| --- | --- | --- |
| `training/` | Optimizers and adapter initialization | [optimizers.py](training/optimizers.py), [qwen35_loftq.py](training/qwen35_loftq.py) |
| `teachers/` | Teacher API requests, resumable annotation, teacher-cache validation | [openrouter_cli.py](teachers/openrouter_cli.py), [openrouter.py](teachers/openrouter.py), [prefix_cache.py](teachers/prefix_cache.py), [prefix_audit.py](teachers/prefix_audit.py) |
| `data/` | Training augmentation and prefix preparation | [judge_injection.py](data/judge_injection.py), [prefix_boundaries.py](data/prefix_boundaries.py), [prefix_sampling.py](data/prefix_sampling.py) |
| `campaigns/` | Concurrent campaign progress | [status.py](campaigns/status.py) |
| `evaluation/` | Scores, metrics, calibration, preference diagnostics, sharding and recovery | [probabilities.py](evaluation/probabilities.py), [binary.py](evaluation/binary.py), [scoring.py](evaluation/scoring.py), [metrics.py](evaluation/metrics.py), [preferences.py](evaluation/preferences.py) |
| `configs/` | Packaged Hydra configuration | [configs/](configs/) |

Use `gleipnir.training.optimizers` for optimizer code. Historical
`from gleipnir.training import MuonAdamW` and the other former public symbols
remain supported through lazy exports. The annotation command is
`python -m gleipnir.teachers.openrouter_cli`; `gleipnir-openrouter` and
`python -m gleipnir.openrouter_cli` remain supported.

## Modules awaiting migration

Serving and kernels remain at their existing paths. Consult the
[root README](../../README.md) for the selected serving and training recipes.

The training runners, backend setup, and campaign preparation helpers also remain
at their existing paths where launchers record their source-file checksums.
Their migration must update those provenance lists to include implementations,
rather than hashing only alias registration. Do not change frozen receipts or
contracts to accommodate a move. Important entrypoints are
`monitoring_systems_screen.py`, `monitoring_campaign_training.py`,
`packed_training.py`, and `branch_trainer.py`.

The source-bound `monitoring_campaign_evaluation.py` orchestrator also remains at
its existing path, exposed as `gleipnir.evaluation.campaign` through the same
alias registry. For new evaluation source fingerprints, use the tuples in
[evaluation/sources.py](evaluation/sources.py). These include moved implementations,
shared score helpers, and alias registration; existing frozen manifests and
their drift checks remain unchanged. Metric definitions, thresholds, raw-logprob
checks, prediction schemas, and contract hashing rules are preserved by the move.
Moving source changes its fingerprints: historical code-bound campaigns continue
to reject drift and require their recorded code revision for reproduction.
