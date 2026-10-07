# Shared source map

Place new shared code in the package matching its responsibility. Historical
import names resolve lazily through the single [_compat.py](_compat.py) registry.
Use canonical package paths for new code; moved implementations have one home.

| Package | Responsibilities | Starting points |
| --- | --- | --- |
| `training/` | Objectives, batching, branch execution, training audits and adapter utilities | [optimizers.py](training/optimizers.py), [packed.py](training/packed.py), [branch_trainer.py](training/branch_trainer.py), [binary_tasks.py](training/binary_tasks.py), [mil.py](training/mil.py) |
| `teachers/` | Teacher API requests, prompt construction and output parsing, resumable annotation and cache validation | [openai.py](teachers/openai.py), [openrouter_cli.py](teachers/openrouter_cli.py), [prompts.py](teachers/prompts.py), [prefix_cache.py](teachers/prefix_cache.py), [prefix_audit.py](teachers/prefix_audit.py) |
| `analysis/` | Shared plotting conventions, Pareto frontiers and empirical scaling fits | [plotting.py](analysis/plotting.py), [scaling.py](analysis/scaling.py) |
| `adapters/` | Adapter artifact transformations with tensor and checksum preservation | [rebase.py](adapters/rebase.py) |
| `data/` | Campaign inputs, augmentation, filtering and prefix preparation | [monitoring.py](data/monitoring.py), [branches.py](data/branches.py), [exclusions.py](data/exclusions.py), [nested_subsets.py](data/nested_subsets.py), [transcript_injection.py](data/transcript_injection.py) |
| `campaigns/` | Training launches, matched systems screens and stage coordination | [systems_screen.py](campaigns/systems_screen.py), [training.py](campaigns/training.py), [training_command.py](campaigns/training_command.py), [lanes.py](campaigns/lanes.py), [status.py](campaigns/status.py) |
| `evaluation/` | Scores, metrics, calibration, preference diagnostics, sharding and recovery | [probabilities.py](evaluation/probabilities.py), [binary.py](evaluation/binary.py), [scoring.py](evaluation/scoring.py), [metrics.py](evaluation/metrics.py), [preferences.py](evaluation/preferences.py) |
| `configs/` | Packaged Hydra configuration | [configs/](configs/) |

Use `gleipnir.training.optimizers` for optimizer code. Historical
`from gleipnir.training import MuonAdamW` and the other former public symbols
remain supported through lazy exports. The annotation command is
`python -m gleipnir.teachers.openrouter_cli`; `gleipnir-openrouter` and
`python -m gleipnir.openrouter_cli` remain supported.

Use `python -m gleipnir.campaigns.systems_screen` for config-driven training
screens. The historical `gleipnir.monitoring_systems_screen` command remains an
alias. Repository/cache roots, job schemas, objectives, and batching are unchanged.

Use `python -m gleipnir.adapters.rebase` to rebase Qwen3.5 adapter keys. The
historical `gleipnir.qwen35_adapter_rebase` command remains an alias.

## Modules awaiting migration

Serving and kernels remain at their existing paths. Consult the
[root README](../../README.md) for the selected serving and training recipes.

Training backend integrations remain flat: `flashqla_training.py`,
`native_fp4_training.py`, `fouroversix_training.py`, `qwen35_fast_training.py`,
`attention_backends.py`, `packed_sequences.py`, `bf16_lora.py`, and
`training_hotpath.py`. Backend benchmark/canary runners also retain their paths.
New source snapshots use canonical implementation paths and include alias
registration. Source lists inherited from a receipt can translate live source
references using `_compat.canonical_source_reference`; archived artifacts and
their recorded hashes are never rewritten.

The source-bound `monitoring_campaign_evaluation.py` orchestrator also remains at
its existing path, exposed as `gleipnir.evaluation.campaign` through the same
alias registry. For new evaluation source fingerprints, use the tuples in
[evaluation/sources.py](evaluation/sources.py). These include moved implementations,
shared score helpers, and alias registration; existing frozen manifests and
their drift checks remain unchanged. Metric definitions, thresholds, raw-logprob
checks, prediction schemas, and contract hashing rules are preserved by the move.
Moving source changes its fingerprints: historical code-bound campaigns continue
to reject drift and require their recorded code revision for reproduction.
