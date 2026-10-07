# Shared source map

Place new shared code in the package matching its responsibility. Historical
import names resolve lazily through the single [_compat.py](_compat.py) registry.
Use canonical package paths for new code; moved implementations have one home.

| Package | Responsibilities | Starting points |
| --- | --- | --- |
| `training/` | Objectives, packing, branch execution, backend integrations, startup validation and training screens | [optimizers.py](training/optimizers.py), [packing.py](training/packing.py), [backends/](training/backends/), [startup.py](training/startup.py), [screens/](training/screens/), [branch_trainer.py](training/branch_trainer.py) |
| `teachers/` | Teacher API requests, prompt construction and output parsing, resumable annotation and cache validation | [openai.py](teachers/openai.py), [openrouter_cli.py](teachers/openrouter_cli.py), [prompts.py](teachers/prompts.py), [prefix_cache.py](teachers/prefix_cache.py), [prefix_audit.py](teachers/prefix_audit.py) |
| `analysis/` | Shared plotting conventions, Pareto frontiers and empirical scaling fits | [plotting.py](analysis/plotting.py), [scaling.py](analysis/scaling.py) |
| `adapters/` | Adapter artifact transformations with tensor and checksum preservation | [rebase.py](adapters/rebase.py), [merge.py](adapters/merge.py) |
| `serving/` | Runtime and compiler identity, CPU frontend, precision audits, engine integration and bundle restoration | [runtime.py](serving/runtime.py), [compile_cache.py](serving/compile_cache.py), [gigatoken.py](serving/gigatoken.py), [vllm/](serving/vllm/), [fp4/](serving/fp4/), [gdn/](serving/gdn/) |
| `kernels/` | Shared GPU arithmetic and model adapters, FP4 packing/GEMMs, NVIDIA MXFP8 attention | [mlp_gemm.py](kernels/mlp_gemm.py), [fp4/](kernels/fp4/), [mxfp8/](kernels/mxfp8/), [nvidia_causal_conv1d.py](kernels/nvidia_causal_conv1d.py) |
| `data/` | Campaign inputs, augmentation, filtering and prefix preparation | [monitoring.py](data/monitoring.py), [branches.py](data/branches.py), [exclusions.py](data/exclusions.py), [nested_subsets.py](data/nested_subsets.py), [transcript_injection.py](data/transcript_injection.py) |
| `campaigns/` | Training launches, isolated runtime setup, matched systems screens and stage coordination | [systems_screen.py](campaigns/systems_screen.py), [runtime.py](campaigns/runtime.py), [training.py](campaigns/training.py), [training_command.py](campaigns/training_command.py), [lanes.py](campaigns/lanes.py) |
| `evaluation/` | Campaign orchestration, scores, calibration, preference diagnostics, sharding and recovery | [campaign.py](evaluation/campaign.py), [binary.py](evaluation/binary.py), [scoring.py](evaluation/scoring.py), [metrics.py](evaluation/metrics.py), [preferences.py](evaluation/preferences.py) |
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

## Source identity

Consult the [root README](../../README.md) for selected serving and training
recipes. Serving integrations live under `serving/`; shared arithmetic lives
under `kernels/`. FP4 compiler, memory and performance diagnostics live under
`training/fp4/`. GPU dependencies stay optional until an implementation is imported.

The package root contains only initialization and alias registration. Backend
integrations live in `training/backends/`; packing, BF16 LoRA, hotpath and startup
helpers live directly under `training/`. Reusable timing and correctness screens
live in `training/screens/`.
New source snapshots use canonical implementation paths and include alias
registration. Source lists inherited from a receipt can translate live source
references using `_compat.canonical_source_reference`; archived artifacts and
their recorded hashes are never rewritten.

The campaign evaluator lives in `evaluation/campaign.py`; the historical
`gleipnir.monitoring_campaign_evaluation` import remains an alias. For new
evaluation source fingerprints, use the tuples in
[evaluation/sources.py](evaluation/sources.py). These include moved implementations,
shared score helpers, and alias registration; existing frozen manifests and
their drift checks remain unchanged. Metric definitions, thresholds, raw-logprob
checks, prediction schemas, and contract hashing rules are preserved by the move.
Moving source changes its fingerprints: historical code-bound campaigns continue
to reject drift and require their recorded code revision for reproduction.
