---
pretty_name: Gleipnir Kimi K3 Tool-Trajectory Logits
task_categories:
  - text-classification
tags:
  - ai-control
  - knowledge-distillation
  - logits
  - monitoring
---

# Gleipnir Kimi K3 tool-trajectory logits

This dataset contains Kimi K3 binary teacher targets for the 8,688-row training
mixture used by Sinha et al., *Monitoring Computer-Use via Hierarchical
Summarization* (arXiv:2605.29601). It supports exploratory logit-distillation
experiments for action-only AI-control monitors.

## Contents

`train.parquet` contains one row per source example and no prompt or trajectory
text. It includes stable source join keys and hashes, the raw returned logprobs
for literal decision tokens `0` and `1`, the normalized binary probabilities,
the generated decision, provider/request provenance, and token-usage fields.
Ground-truth labels are intentionally omitted; obtain the upstream artifact and
join it using the pinned source revision and row identity fields.

`metrics.json` reports aggregate diagnostics computed locally before labels were
removed. `manifest.json` records the exact private-cache checksums, public-file
checksums, model/provider counts, prompt-template hash, and request-setting
hashes.

## Teacher contract

- Model: `moonshotai/kimi-k3` through OpenRouter.
- Interface: terminal `Prediction:0` or `Prediction:1` with logprobs for both
  literal decision tokens.
- Temperature: zero; reasoning effort: `none`; top logprobs: five;
  `max_tokens=64`; no provider fallback.
- Prompt template SHA-256:
  `2418cc55801deead8983d3bde6e35b59603f0080711ce8e12e1d7299487ac128`.
- Provider shards: 5,071 Makora rows, 305 Morph rows, and 3,312 Fireworks rows.

The provider handoffs were validated on 64-row overlap canaries. Provider is
retained as a column and should be used for stratified diagnostics rather than
assuming exact logit interchangeability.

## Training-mixture diagnostics

The complete cache has pooled AUROC `0.9673` and raw normalized pooled
pAUROC@20 `0.8645`. The unweighted mean across the five source datasets is
AUROC `0.9914` and pAUROC@20 `0.9636`. These are in-distribution training-data
diagnostics, not held-out or OOD benchmark results.

## Source and license warning

The source artifact is
`adityaasinha28/deliberative-monitor-pipeline`, revision
`2f10f4e49f4bf349965640ac021406689455d1c3`, file
`merged/stride_gloom_crh_s45_o46_ctrl_t7_bal.parquet`. Its repository did not
declare a license when this cache was created. This dataset therefore does not
redistribute source prompts, trajectories, or labels and does not grant rights
to the upstream data. Users must obtain the source artifact themselves and
determine that their use complies with its terms and the terms of its component
datasets.

The teacher annotations and metadata are published for research use without a
separate license grant. Provider outputs may also be subject to the applicable
model-host and inference-provider terms.

## Reproducibility

The corresponding Gleipnir finding records the collection history, cost and
cache telemetry, provider-agreement canaries, checksums, metrics, and caveats:
`docs/findings/tool_trajectory_training_teacher_cache.md` in the Gleipnir
repository.
