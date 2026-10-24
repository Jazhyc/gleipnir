# Activation probes versus LoRA monitors

User direction, 2026-10-24: measure how probes on frozen Qwen3.5-4B activations
generalize compared with soft-distilled LoRA monitors (Gleipnir 4B), and
whether Kimi K3 soft targets help probes the way they help LoRA training.
Status: design only. No capacity is reserved and nothing has run.

## Hypotheses

1. **Objective.** Kimi soft targets give probes higher OOD pAUROC@20 than hard
   source labels, with the backbone and architecture held fixed.
2. **Backbone prompt.** Probes on activations computed under the full teacher
   prompt beat probes on plain next-token-prediction (NTP) activations. A
   pooling probe recovers part of that gap, because only pooling can gather
   evidence that the prompt would otherwise bring to the readout position.
3. **Probe versus LoRA.** The best probe's ID-to-OOD drop is no larger than
   Gleipnir 4B's. This is descriptive, not a promotion rule.

Prior evidence: on the 27B, a linear hidden-state probe at the decision
position lifted the unadapted model from 0.8869 to 0.9117 ID pAUROC@20,
compared with 0.9456 for trained Gleipnir 27B under the same prompt
([vocabulary diagnostic](../gleipnir27b_vocab_distillation/README.md)). OOD
was never measured for the base probe.

## Design

A 2×2 grid, crossed with a probe-architecture sweep.

- **Backbone.** Unadapted `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`,
  BF16 forward.
  - *Prompted*: the full teacher prompt rendered as the frozen benchmark renders
    it: chat template, thinking off, `Prediction:`. For ID this is the
    canonical CoT-removed trajectories under the teacher template
    (`teacher:id_cot_removed` in the vocabulary diagnostic).
  - *NTP*: the bare trajectory text from `trajectory_from_prompt`, with no
    instructions, chat template or suffix.
- **Objective.** BCE with a bias term against (a) the source hard label or
  (b) Kimi K3's `soft_target` from `data/qwen38-27b-s0-01/strict_filter/teacher.jsonl`.
- **Rows.** The 7,836 strict-filtered training rows (the Gleipnir 4B/27B
  population), the 3,012 canonical CoT-removed ID rows, and the frozen 6,395-row
  OOD suite. Training and ID lineages stay disjoint, as in the existing
  preparation.
- **Activations.** The residual stream after decoder block 15 (zero-based; the
  middle of 32 layers, hidden size 2560). Verify both numbers against the pinned
  config before extraction. Store only trajectory tokens, plus the suffix after
  `</agent_trajectory>` in the prompted condition. The instruction prefix is
  identical across rows, and causal attention makes its activations identical
  too, so it carries no row information. Also store the last token of every
  layer in BF16 (about 3 GB) for the layer sweep of the last-token probe.
- **Probes.** All except the last-token probe follow
  [Kramár et al. 2026](https://arxiv.org/abs/2601.11516). The per-token
  transform is a 2-layer ReLU MLP of width 100.
  - last-token logistic regression, with a layer sweep;
  - mean pooling, with a linear readout and with the per-token MLP;
  - softmax attention probe with 10 heads;
  - MultiMax: a hard max per head, summed over heads;
  - max of rolling attention means, w ∈ {10, 64, 256}.

  Optimizer defaults are AdamW at LR 1e-4 and weight decay 3e-3. Minibatches
  replace full-batch training at a fixed step budget, which the pilot freezes.
  Train every probe of a cell from one shared minibatch stream, so each read of
  the activation store serves all architectures and seeds.
- **References (no new runs).** Gleipnir 4B v0.4 BF16 (ID 0.9335, OOD 0.804),
  and the base 4B zero-shot teacher-prompt margin (OOD 0.6175, archived at
  `results/tool_trajectory_monitoring/qwen35_4b_teacher_ood/`). The ID margin
  falls out of the prompted extraction for free.

Out of scope, by user decision: probes on Gleipnir activations, Gleipnir in NTP
mode, and suffix-only prompting.

## Selection and stop rules (frozen)

- In each of the four cells, select architecture, window, step budget and
  seed on ID source-macro raw normalized pAUROC@20. Break ties by AUROC, then
  by lower macro Brier.
- Read OOD once per cell for the selected probe, and once for each
  architecture's ID-best configuration (descriptive). Report every candidate's
  ID result, plus per-source ranking, calibration, thresholds and ties.
- Hypothesis 1 holds per backbone if soft beats hard on Mean-OOD pAUROC@20 by at
  least 0.01, using the median of 5 seeds at the selected configuration.
  Hypothesis 2 is measured the same way across backbones.
- Stop and report if extraction parity, the FP8 gate or the storage audit
  fails. Never select on OOD.

## Pilot (before the full extraction)

The pilot freezes the activation precision and the step budget, and gives a
first look at how the architectures differ.

- **Rows.** 1,000 training rows and 1,000 ID rows, sampled by lineage group
  and stratified by source and label, under both backbones. That is about
  19M trajectory tokens per backbone.
- **Parity.** On 128 OOD rows, the decision margin from the prompted forward
  (final norm, then LM head) must match the archived stock-vLLM base-4B
  predictions with correlation ≥ 0.999. Record the MAE. These are score
  checks against an existing result, not probe evaluation.
- **FP8 gate.** Store BF16, then derive E4M3 FP8 offline with one FP32 scale per
  token per 128-channel block. Per-block scales keep residual-stream outlier
  channels from crushing the rest. For each architecture:
  - (a) A probe trained on BF16 must score the FP8 copy with logit
    correlation ≥ 0.999 and |Δ macro pAUROC@20| ≤ 0.005.
  - (b) Probes trained on FP8 and on BF16 with the same seeds must differ in
    median macro pAUROC@20 by at most 0.005.

  If FP8 fails, stop and ask the user whether to store BF16 (about 1.6 TB).
- **Budget.** Sweep step counts on the pilot ID rows and freeze the full-run
  budget from that sweep. Pilot architecture results are descriptive; they
  may drop an architecture only if it is clearly dominated on every pilot cell.

## Storage and compute

Trajectory text totals about 517M characters across train, ID and OOD,
roughly 152M tokens per backbone at the measured 3.4 characters per token. One
layer at FP8 is about 0.39 TB per backbone, or **about 0.8 TB for both**,
plus about 3% for scales. The pilot adds about 0.3 TB (BF16 plus FP8). The
extraction audit replaces these estimates with exact token counts.

Planned target: a Lambda GPU instance (one H100-class GPU is enough for a 4B
forward at 30K tokens) with a persistent filesystem of about 1.2 TB in the same
region. Lambda filesystems are tied to their region, so check GPU availability
there first. Probe training is I/O-bound; stage the active backbone's store to
local NVMe if the filesystem is slow. Launching or terminating capacity needs
the user's explicit instruction. Pull probe weights, predictions and receipts
to `results/probe_generalization/` before any termination.

Artifacts: `data/probe_generalization/` (manifests and row samples),
activation stores on the Lambda filesystem, `results/probe_generalization/`,
and logs under `logs/lambda/probe_generalization/`.
