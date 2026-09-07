# `gleipnir-improvement` shutdown inventory

Date: 2026-09-08 (Europe/Amsterdam). Status: artifacts collected and verified;
termination explicitly authorized and confirmed. The two-H100 instance is no
longer active.

The audited target is instance `bdd7ae2cf92c4f4386ee307972553d60`,
`gpu_2x_h100_sxm5`, in Lambda `us-south-2`. Both H100 80GB SXM5 GPUs reported
0% utilization and 0 MiB allocated, with no active training, evaluation, vLLM,
or campaign supervisor found. No attached persistent filesystem was listed.
The final GPU check again found no compute processes; the Lambda API still
listed the instance as active at $8.38/hour.
The remote result tree occupied about 160 GB. The selected artifacts below
preserve the completed scientific endpoints without copying that whole tree.

## Missing artifacts collected

The initial comparison found missing final ablation weights, 79 prediction files,
77 result files, corrected rationale-run evidence, and two derived training
inputs. Several experiment READMEs still described completed work as pending;
their completion records now point to the final results.

The metadata archive contains 2,581 files: 2,066 previously missing data, result,
and log files installed at their original repository-relative paths; nine remote
versions of existing artifacts retained without overwriting local versions;
and a 506-file source/configuration/documentation snapshot. The nine differing
artifacts are enumerated in `metadata_verification.json`. They include status
files and systems-screen manifests; use the archived remote version when
reconstructing its runtime context.

The remote checkout was a synced snapshot with no `.git` directory. Its source
snapshot is preserved separately in the archive, alongside per-run source hashes
and commit metadata, rather than replacing current repository code. It captures
the final remote checkout; individual runs may record earlier source revisions.

All generated archives, weights, predictions, caches, inputs and logs are ignored
by Git. The audit directory is
`results/lambda_improvement_shutdown_audit/`.

| File in audit directory | Bytes | SHA-256 |
|---|---:|---|
| `research_artifacts.tar` | 1,253,959,680 | `d91392d428eb4838f54ba8c6f7b7a444eb7ce80990abd42ff41596505c132797` |
| `scientific_masters.tar` | 21,064,929,280 | `7a3f41477e90a1de9081662a6800d54b068e45ecb16b3c81606214507d7fecf0` |
| `verification_manifest.json` | 706,307 | `9b21699b998631612682a4b06f7575bc2c5d025e6d6f269eeac504a193ae4bdf` |

The manifest records every selected file's path, size and full SHA-256, plus
tensor counts and dtypes for new master weights. Both local archives match the
remote archive identities; all 2,581 metadata members and all 31 master members
passed individual hash checks. Local archive/member verification is recorded in
`metadata_verification.json` and `masters_verification.json`. Temporary transfer
chunks were removed only after successful archive verification.

## Final scientific masters

Retain all 31 new final causal-LM masters, including negative outcomes, so later
input-contract evaluations do not require retraining. Each contains 256 FP32
tensors and is 679,511,752 bytes. Their adapter configurations, training metadata,
job configurations, provenance, final trainer states and available serving-parity
reports are retained locally. Thirty ordinary trainings reached their configured
final update; the branching run reached its predeclared compute stop condition.

| Result tree under `results/` | Masters |
|---|---:|
| `monitoring_lr_sweep` | 5 |
| `monitoring_duration` | 2 |
| `monitoring_id_scaling` | 4 |
| `monitoring_objective_ablation` — valid MIL arms | 3 |
| `monitoring_objective_ablation_accumulation_v2` — corrected rationale arms | 2 |
| `monitoring_prefix_training` | 2 |
| `monitoring_prefix_training_low` | 2 |
| `monitoring_subset_duration` | 9 |
| `monitoring_mil_mixture_ddp` | 1 |
| `monitoring_branching_compute1x` | 1 |

The existing mixed 4B, mixed 9B and monitoring-only 9B masters were independently
content-hashed locally and on Lambda and match the identities in the
[previous campaign inventory](gleipnir_control_shutdown_inventory.md).
Their fresh comparison is in `additional_verification.json` and
`baseline_verification.json`.

## Data, teachers and runtime provenance

All 454 previously present, selected nonweight artifacts match Lambda by full
SHA-256, including cleaned ID inputs, evaluations, manifests and the prefix
teacher cache (133,947 records, matching its completion marker). The two
previously missing derived inputs are now installed at
their canonical paths. Important identities are:

| Artifact | SHA-256 |
|---|---|
| `data/id_cot_only_evaluation/prompts.jsonl` | `7a4ec604740b4b3b7bd70aafd4473b1b6bfcbf92706bcd23b90ed8df2c4bc1b4` |
| `data/id_cot_only_evaluation/evaluation.manifest.json` | `abcc8267a77f6eff0a252ea846a6a59283e8a027cc860210e3984064eec611f1` |
| `results/monitoring_prefix_supervision/qwen35_flashinfer_cache/logits.jsonl` | `024d6e2c6c9e81aa2ea979d6ebe3f267e71e5c8a8ce3e67e4410ee7f96cb32a1` |
| `data/monitoring_prefix_supervision/training/paired_seed0.jsonl` | `50ae0582ec078a9fe9574ce436a877a19aacd51cd6be1d5217432816362d6d6b` |
| `data/tool_trajectory_monitoring/distillation_scaling/rationale_student_rows.jsonl` | `8d521597283196741f423320d78a59eec92e6029bc0c87a97a5de4678ccfc7b1` |

Historical Kimi training caches already reside locally: 5,071 original-provider,
3,312 Fireworks and 305 Morph records, totaling 8,688. Their full hashes are in
`historical_teacher_caches.json`. Those caches were not present in this Lambda
snapshot, so this is a local inventory rather than a fresh remote comparison.
Preserve raw responses and prompt/provenance fields alongside derived targets.
The prefix cache's documented numerical limitations remain in force; a matching
backup hash does not validate its teacher numerics.

`additional_verification.json` records the two installed environment versions.
Per-run metadata is authoritative for actual training kernels and settings,
including isolated FLA/causal-conv1d, quantization, batch, memory and throughput.
The archived lockfile and source snapshot preserve the environment definitions.
Base-model caches, virtual environments and compiled-kernel caches can be rebuilt.

## Scientific closure and remaining work

- [Corrected rationale SFT](monitoring_objective_accumulation.md) is complete.
  Its two weights score `0.870399` and `0.873341` historical-ID pAUROC@20;
  neither clears the frozen gain threshold over soft-only `0.871066`.
- [All nine subset-duration endpoints](../../experiments/monitoring_subset_duration/README.md)
  are complete. Three-epoch MIL reaches `0.895951` historical-ID pAUROC; every
  sampled-prefix endpoint is below its matched-duration soft-only control.
- [Adding the full deception pool](../../experiments/monitoring_mil_mixture/README.md)
  reduces selected-MIL ID pAUROC to `0.876761` under the tested mixed recipe.
- [The selected MIL OOD comparison](../../experiments/monitoring_mil_ood/README.md)
  is complete: `0.781299` versus released 4B `0.782350` macro pAUROC. It does
  not establish an OOD improvement, and the comparison changes multiple factors.

The main unperformed evaluation is the newer ablation adapters on the
CoT-removed ID contract. Historical-ID rankings must not be treated as validated
rankings on that input contract. The retained masters allow this later on local
Slurm, with the usual master/export/backend parity checks and a frozen candidate
and selection rule. Further training on cleaned inputs, consistency objectives,
compact evidence targets, and checkpoint averaging remains the
[documented research backlog](../research/monitoring_training_followups.md).
Those proposals do not require keeping this instance allocated and were not
launched by this audit.

Completion checks also verified the expected prediction count, unique
`(source, id)` pairs and finite probability scores in all 16 final evaluations
from the subset-duration, corrected-objective, mixed-MIL and MIL-OOD runs.
`completion_checks.json` records their final metrics and coverage checks.

## Deliberate exclusions and recovery

Excluded weights are systems/throughput canaries, abandoned preflights, the two
invalid original rationale arms, intermediate/final-step duplicate checkpoints,
and reproducible serving copies. Their available configurations, logs and
diagnostic metrics are retained. Optimizer/RNG states, compiler caches, base
weights and repeated tokenizer payloads are not included. This preserves final
scientific endpoints for evaluation; it does not promise bitwise continuation
of discarded training states or recover intermediate checkpoints for averaging.

Weights remain archived to avoid a second 21 GB local copy. To recover one master
from the repository root, extract its exact manifest path, for example:

```bash
tar -xf results/lambda_improvement_shutdown_audit/scientific_masters.tar \
  results/monitoring_subset_duration/runs/mil-pct020-lr2em05-epochs3-seed0/causal_adapter/adapter_model.safetensors
```

Its configuration and training metadata are already at that path's parent.
Verify the member's manifest checksum, restore the tokenizer from the pinned base
revision, regenerate the serving adapter, and pass the matched parity gate before
evaluation. To inspect the final remote source or an alternate manifest without
overwriting current files, extract `research_artifacts.tar` into a separate empty
directory. These recovery steps use retained local artifacts and pinned upstream
dependencies.

## Termination confirmation

After the user explicitly authorized termination on 2026-09-08
(Europe/Amsterdam), `scripts/lambda_cloud.py terminate --campaign
gleipnir-improvement --yes` submitted the request for the audited instance
`bdd7ae2cf92c4f4386ee307972553d60`. Lambda accepted it and reported
`terminating`. Follow-up API observations are retained locally in
`results/lambda_improvement_shutdown_audit/termination_checks.jsonl`.

At 2026-09-07 23:25:56 UTC (2026-09-08 01:25:56 Europe/Amsterdam), the
campaign-filtered instance query returned no matching instance. At 23:26:16 UTC,
the full Lambda instance query also returned an empty list, confirming that the
two-H100 capacity was released. No replacement capacity was launched. The
checksummed local artifacts listed above were retained before termination.
