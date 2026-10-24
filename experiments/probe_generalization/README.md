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
    instructions, chat template or suffix. Qwen has no BOS token, so an NTP
    row's first token is the attention sink at sequence position 0. Pooled
    probes drop it (user decision, 2026-10-24); the prompted span never
    contains it.
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
  - max of rolling attention means, w ∈ {10, 64, 256};
  - the best AlphaEvolve-discovered probe (the paper's Algorithm 1 and Table 3
    "AlphaEvolve" row): LayerNorm, the same MLP, Softplus-gated projections,
    bipolar max/−min pooling and a linear readout, trained with an L1 penalty
    on all weights and an orthogonality penalty on the projection. The paper
    omits its channel count and penalty weights; we use 10 channels (its
    default head count) and record the weights in the probe name. Added after
    the pilot, so it enters at the full run.

  Optimizer defaults are AdamW at LR 1e-4 and weight decay 3e-3. Minibatches
  replace full-batch training at a fixed step budget, which the pilot freezes.
  Train every probe of a cell from one shared minibatch stream, so each read of
  the activation store serves all architectures and seeds.
- **References (no new runs).** Gleipnir 4B v0.4 BF16 (ID 0.9335, OOD 0.804),
  and the base 4B zero-shot teacher-prompt margin (OOD 0.6175, archived at
  `results/tool_trajectory_monitoring/qwen35_4b_teacher_ood/`). The ID margin
  falls out of the prompted extraction for free.

Out of scope, by user decision: probes on Gleipnir activations, Gleipnir in NTP
mode, and suffix-only prompting. Architecture search is also deferred (user
decision, 2026-10-24): this study keeps the Kramár et al. family fixed and
varies the objective (hard labels versus Kimi soft targets), which no prior
probe work tests. A later headline study may search architectures at matched
probe compute budgets, with the paper's AlphaEvolve probe (Appendix F.2) as
the baseline to beat.

## Selection and stop rules (frozen)

- In each of the four cells, a candidate is an (architecture, checkpoint
  epoch) pair trained with 5 seeds; architectures include each rolling window
  and each AlphaEvolve penalty setting. Its ID score is the seed-median
  source-macro raw normalized pAUROC@20 on the full canonical CoT-removed ID
  set, ties broken by seed-median AUROC, then lower seed-median macro Brier.
  The best candidate is the cell's selection.
- Read OOD once per cell for the selected candidate (all 5 seeds; report the
  seed median and the best-ID seed), and once for each architecture family's
  ID-best candidate (descriptive; the AlphaEvolve grid is one family). OOD
  logits of every other probe are discarded unread. Last-token probes select
  (layer, C) per objective on ID. Report every candidate's ID result, plus
  per-source ranking, calibration, thresholds and ties.
- Full-run settings (frozen 2026-10-24, before any full extraction): FP8
  storage only; 7,836 training rows, batch 16 rows, AdamW at LR 1e-4 and
  weight decay 3e-3; checkpoints at epochs 1, 2, 4 and 8 (about 490 to 3,920
  updates, spanning the pilot's best 1,008); a 65,536-token cap per forward with
  gradient accumulation (same updates). Implementation: `run_full.sh`,
  `evaluate_ood.py` (selection, then OOD).
- Hypothesis 1 holds per backbone if soft beats hard on Mean-OOD pAUROC@20 by at
  least 0.01, using the median of 5 seeds at the selected configuration.
  Hypothesis 2 is measured the same way across backbones.
- Stop and report if extraction parity, the FP8 gate or the storage audit
  fails. Never select on OOD.
- Budget extension (user decision, frozen 2026-10-24 before any epoch-8
  result existed): if a cell's selected candidate is at the last checkpoint
  (epoch 8), that backbone's sweep continues to epoch 16 with a checkpoint
  there, and selection reruns over epochs {1, 2, 4, 8, 16} before any OOD
  read. Sweeps that saved AdamW state resume exactly; the prompted sweep
  started before optimizer checkpoints existed, so it resumes from its
  epoch-8 weights after rebuilding AdamW moments by replaying epoch 8's
  batches at learning rate 0 (weights verified unchanged), which avoids the
  sign-step kick of fresh moments; a recorded deviation. Epoch 12 is also a
  checkpoint in every extension, so candidates span {1, 2, 4, 8, 12, 16}
  (user decision after the prompted epoch-8 ID read, before any OOD read;
  the prompted extension restarted from its epoch-8 state to include it).

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
  *Result:* correlation 0.99890, MAE 0.070, max 0.27 logit (reference SD
  1.84), so the strict gate **failed**. A BF16 head does not close it (0.99874),
  so the gap is Transformers/FLA/SDPA versus vLLM 0.31 kernels in the forward
  pass, not head precision. The user accepted it as unnoticeable
  (`user_accepted_failed_parity`, 2026-10-24).
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
- **Settings.** Every sweep probe trains at batch 16 rows with 3 seeds; seeds
  change only initialization, and hard/soft probes of one seed start from the
  same weights. Checkpoints at epochs 2, 4, 8 and 16 (about 1,000 updates at
  16 epochs, matching the paper). Last-token probes sweep all 32 layers and
  C ∈ {0.001, 0.01, 0.1, 1}. MultiMax takes each head's maximum value
  (Eq. 9, no query vector), both in training and in scoring.

Code: `prepare.py` (rows), `extract.py` (activations), `train.py` (pooled
sweep), `lasttoken.py`, `pilot.py` (gates) and `run_pilot.sh` (remote order);
reusable parts live in `src/gleipnir/probes/`, tests in `tests/probes/`.
Each sweep trains as one fused `ProbeBank` (all first layers in
one matmul, one elementwise AdamW over disjoint parameters, so each probe
trains as if alone), with the training rows resident in GPU or pinned CPU
memory and evaluation rows in pinned CPU memory. Every probe scores both
formats, so each sweep also reports cross-precision transfer. The
pilot restarted on this trainer (2026-10-24 11:04); its first prompted FP8
sweep started before the evaluation change.

## Pilot result (2026-10-24, `pilot01`)

Completed 12:06 UTC on one Lambda H100 PCIe. Artifacts:
`results/probe_generalization/pilot01/` (summaries, logits, probe weights,
`pilot_summary.json`), logs under `logs/lambda/probe_generalization/pilot01/`,
stores on the `gleipnir-probes` filesystem. Descriptive only: 1,000 ID rows,
which also choose the budget.

- **Parity:** failed strictly, user-accepted (see above).
- **FP8 gate (a): passed** on both backbones (worst logit correlation 0.9999 /
  0.99976, worst |Δ pAUROC| 0.0015 / 0.0031).
- **FP8 gate (b): failed** on both (4 / 8 of 56 cells; worst |Δ| 0.0126 /
  0.0199). All failures are MultiMax or rolling_w10. Signed Δ averages −0.0003
  (prompted) and +0.0009 (NTP), per-seed differences change sign, and these
  probes' BF16 seed range reaches 0.11–0.19, so the failures read as training
  divergence of argmax-routed probes rather than storage error. Mean |Δ| is at
  most 0.0017 for rolling w64/w256 and attention. The user accepted FP8
  storage for the full run (`user_accepted_failed_fp8_gate_b`, 2026-10-24),
  judging the shift minimal.

Seed-median ID pAUROC@20 at epoch 16 (FP8-trained, FP8-scored), hard / soft:

| probe | prompted | NTP |
|---|---|---|
| mean_linear | 0.508 / 0.502 | 0.490 / 0.492 |
| mean_mlp | 0.568 / 0.573 | 0.550 / 0.561 |
| attention | 0.800 / 0.802 | 0.830 / 0.823 |
| multimax | 0.733 / 0.773 | 0.727 / 0.768 |
| rolling_w10 | 0.755 / 0.741 | 0.749 / 0.771 |
| rolling_w64 | 0.799 / **0.848** | 0.831 / 0.825 |
| rolling_w256 | 0.804 / **0.848** | **0.857** / 0.851 |
| last token (best layer) | 0.747 / 0.755 (layer 17) | 0.406 / 0.403 (layers 24 / 22) |
| zero-shot margin | 0.6725 | — |

Most pooled probes still improve from epoch 8 to 16 (≈500 → 1,000 updates).
Soft targets lead by about 0.04–0.05 on prompted rolling w64/w256 and
MultiMax, mostly on Gloom, while hard-label training loss falls to about 0.005
(memorization); on NTP the objectives tie except MultiMax. Pooled probes on
NTP activations match the prompted ones although the NTP last token carries
little signal, so pooling recovers what the prompt would surface. FP8-trained
probes score BF16 activations within 0.001.

## Full-run result: prompted backbone (2026-10-24, `full01`)

Extraction, FP8 storage and the 16-epoch extension (triggered: both cells
selected epoch 8; moments rebuilt at lr 0) completed; OOD was read once, at
18:31 UTC. The zero-shot margin reproduces the archived benchmark on OOD
(0.6162 vs 0.6175).

| cell | ID selection | ID | OOD (5-seed median) |
|---|---|---:|---:|
| hard | attention, epoch 12 | 0.8434 | 0.7057 |
| soft | rolling_w10, epoch 8 | 0.8564 | 0.7126 |

Soft − hard = +0.0069 OOD, below the frozen 0.01 bar: **hypothesis 1 is not
met on the prompted backbone.** Per family (each family's ID-best candidate,
descriptive), soft is ahead in 7 of 9 including the last-token probe
(rolling_w256 0.763 → 0.783, multimax 0.721 → 0.755, attention 0.706 → 0.723,
AlphaEvolve 0.696 → 0.706, mean_mlp 0.548 → 0.600, last token 0.716 → 0.727)
and behind for rolling_w64 (0.755 → 0.748) and rolling_w10 (0.740 → 0.713).
ID ranking transfers poorly: the ID winners are among the weaker OOD probes,
while soft rolling_w256 (fourth on ID) is the best OOD family at 0.783,
against Gleipnir 4B's 0.804. SHADE-Arena stays near 0.23–0.25 for every probe;
OOD Brier is 0.23–0.30. Results: `results/probe_generalization/full01/`.

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
