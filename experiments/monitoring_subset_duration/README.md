# Duration, MIL and sampled prefixes on the frozen 20% subset

Hypothesis: repeated exposure to the selected 1,738 monitoring parents may
improve generalization, and weak action-level supervision may change that
duration dependence. User authorized six fresh seed-0 runs: standard soft-only
and best historical MIL, each at 2, 3 and 5 epochs. No warm starts, intermediate
checkpoint selection, new teachers, rationale CE, prefix training or strict OOD.

Use the exact existing nested seed-0 20% selection, not a newly drawn sample.
The best prior MIL pool by ID macro pAUROC@20 was log-mean-exp (0.87612093),
versus top-three mean (0.87577597) and max (0.86849471). Preserve its weight
0.25, temperature 1 and at most eight deterministic action endpoints. Each
bag uses the same full-trajectory Kimi soft target; there are no prefix labels.
MIL is weak localization, not contrastive training. The standard loss remains
the full-boundary Kimi soft BCE with weight 1. Neither objective uses hard labels.

Preserve LR 2e-5, linear decay with 3% warmup, ordinary AdamW, no weight decay,
rank128/alpha256 NF4 double-quantized QLoRA/BF16, FP32 master adapters,
microbatch1/accum32 and the pinned base/kernels. Fresh schedules have 110,
165 and 275 optimizer updates respectively. The MIL job reuses its original
rationale-enriched artifact only as a container: completion supervision is off.
Preparation verifies identical selected direct inputs against the standard rows.

Preserve each previously validated systems recipe: standard uses linear-only
checkpointing and full-attention/linear-shell compilation; MIL uses all-block
checkpointing and linear-shell-only compilation with required flash SDPA.
Record this systems difference rather than claim numerically identical execution.
Both H100 80GB SXM5 GPUs on the existing Lambda target were idle at preparation.
One lane per objective runs 2, then 3, then 5 epochs. Each lane first passes a
longest-32 selected-row accumulation/compile canary. After its three trainings,
it runs master/export serving parity and one persistent TP1 vLLM engine for its
three final adapters. Thus both GPUs can independently train/evaluate; no DDP.

Report all six endpoints on the unchanged 3,012-row ID suite, alongside the
20% one-epoch baseline (pAUROC .88666977, Brier .07553943). Report paired MIL
minus standard differences at each duration, source metrics, calibration,
threshold diagnostics, ties and runtime. The existing +.005 pAUROC/no >.01
source loss/no >.005 Brier regression gate is exploratory only. The subset and
MIL pool were already selected on ID, so this is further development, not an
independent confirmation or automatic promotion.

Fail closed on artifact/selection drift, unavailable kernels, nonfinite loss,
preflight failure, metadata drift, serving mismatch or incomplete evaluation.
Keep all negative outcomes. A failure stops its lane; it never kills the other
lane or unrelated work. The execution queue is not an agent heartbeat: this
session has no scheduling tool for automatic agent follow-ups after turn end.

```bash
PYTHONPATH=src .venv/bin/python -m experiments.monitoring_subset_duration.run prepare
PYTHONPATH=src .venv/bin/python -m experiments.monitoring_subset_duration.run run
```

Logs: `logs/lambda/monitoring_subset_duration/`; frozen contracts, per-lane
status, checkpoints and results: `results/monitoring_subset_duration/`.

## Authorized second stage

After **all six first-stage ID evaluations succeed**, run three fresh models
at 2, 3 and 5 epochs on the same 20% parents with the best previous one-sampled-
prefix recipe (weight .1; previous full-data ID pAUROC .876688). No adaptive
decision or new weight search is implied by this dependency. GPU0 takes five
epochs; GPU1 takes two then three, balancing five epochs of work per device.
Both lanes then run serving parity and their final ID evaluations. Summarize
all nine endpoints and retain the first-stage summary separately.

Reuse the exact seed-0 paired artifact, including the same fixed sampled prefix
for each eligible parent across epochs. Do not resample or recache. The loss is
`(BCE(Kimi_full) + .1 * BCE(Qwen_prefix)) / 1.1` for eligible parents; parents
without a prefix retain their unscaled full BCE. Parent counts, cache contract
and the original authorized failed-numerical-audit exception are recorded.
Prefix targets remain numerically imperfect and mixed-teacher calibration is
an unresolved limitation. The prefix systems recipe matches the standard lane.

## Launch record (2026-09-07)

Supervisor 194727 owns the nine-job queue on `gleipnir-improvement`. Both
longest-32-row compile/accumulation preflights passed, and the initial two-epoch
trainers advanced to optimizer steps 3/110 (standard) and 2/110 (MIL) at the
startup handoff. All 21 focused tests passed locally and on Lambda. The exact
20% selection SHA256 is
`ddb89519df6d4b124b2544536993c3a0f1055030ec63a3a446b958a0f201139a`.
Its 1,270 prefix-eligible parents retain the original fixed sampled targets.
Frozen manifests and all four lane evaluation configs were collected locally.
No timed agent heartbeat was scheduled; the remote queue is execution only.
