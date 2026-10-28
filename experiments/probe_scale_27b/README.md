# Probe versus LoRA at 27B

User direction, 2026-10-27: test whether activation probes stay competitive
with LoRA-trained monitors beyond the 4B. This uses the NTP backbone only, on
one authorized Lambda GPU with the `gleipnir-probes` filesystem. The 4B
results this builds on are in [the probe study](../probe_generalization/README.md).

A 1,000-row pilot was frozen first (`a0e3b72`) and dropped before any
extraction, by user decision. At the 4B, soft targets pulled ahead only with
full training data, and that regime is where probes approach LoRA. A pilot
would have measured the regime where probes don't compete.

## Question

At the 4B, the selected NTP soft probe matches the LoRA seeds on ID (0.904 vs
a 0.903 seed mean) and trails them on OOD (0.764, against seeds at
0.777–0.813). Does the probe–LoRA gap shrink, hold or grow at 27B, where
Gleipnir 27B (seed 0, `qwen38-27b-s0-01`) reaches 0.9217 ID and 0.9087 OOD
pAUROC@20?

## Design (frozen 2026-10-27, before any 27B extraction)

The 4B `full01` NTP protocol, with only the backbone changed:

- **Backbone.** Unadapted `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
  (64 blocks, hidden 5,120), BF16 Transformers forward, NTP rendering (bare
  trajectory, first token dropped as the attention sink). Residual stream
  after block **31** (zero-based), the same relative depth as the 4B's block
  15 of 32 (user choice). Extraction checks the block count and hidden size
  at load.
- **Rows.** `data/probe_generalization/full/`, unchanged: the 7,836
  strict-filtered training rows with Kimi K3 soft targets, the 3,012 canonical
  CoT-removed ID rows and the frozen 6,395-row OOD suite.
- **Probes and training.** As in `full01`: the eleven architectures
  (including the AlphaEvolve penalty grid), hard and soft objectives, seeds
  0–4, batch 16, AdamW at 1e-4 / 3e-3, a 65,536-token forward cap,
  checkpoints at epochs 1, 2, 4 and 8, and FP8 storage. Last-token probes
  sweep all 64 layers.
- **Selection and OOD read.** The frozen rule and budget extension of
  `evaluate_ood.py`, unchanged. Selection uses the seed-median ID pAUROC@20,
  then AUROC, then lower Brier. A cell selected at epoch 8 extends the sweep
  to epoch 16 (checkpoints 12 and 16) before OOD is read. OOD is read once,
  for each cell's selection and each family's ID-best candidate. Nothing is
  selected on OOD.

## Comparisons (descriptive, computed locally afterwards)

1. **Probe–LoRA gap at each scale.** Compare the 27B probe with Gleipnir 27B
   on Mean-OOD and per source, and set it beside the 4B probe–LoRA gap.
2. **Backbone effect.** Compare the 27B probe with the 4B `full01` probe, per
   objective and family.
3. **Soft versus hard at 27B,** measured the way hypothesis 1 was measured at
   the 4B (≥ 0.01 Mean-OOD at the seed median).

Gleipnir 27B is a single seed. The 4B LoRA OOD seed SD is about 1.8 points.

## Results (2026-10-28)

The run finished at 01:10 UTC (`FULL_DONE`). Both objectives selected rolling
attention w256 at epoch 1, so the budget extension did not run. OOD was read
once. Seed-median source-macro pAUROC@20:

| Monitor | ID | OOD |
|---|---:|---:|
| 27B probe, soft targets (selected) | 0.9357 | **0.8481** |
| 27B probe, hard labels (selected) | 0.9099 | 0.8142 |
| Gleipnir 27B, student prompt (frozen reference) | 0.9217 | 0.9087 |
| Gleipnir 27B, teacher prompt (default since 2026-10-20) | 0.9420 | 0.9352 |
| Unadapted Qwen3.8-27B, teacher prompt, zero-shot | — | 0.8901 |
| 4B probe, soft / hard (`full01`) | 0.9036 / 0.8530 | 0.7643 / 0.7092 |

1. **Probe–LoRA gap.** The 27B probe beats Gleipnir 27B on ID but trails it on
   OOD by 0.061 under the frozen student-prompt reference (0.087 under the
   teacher prompt). At the 4B the soft probe sat 0.013–0.049 below the three
   LoRA seeds, so the gap holds or grows with scale. It also stays 0.042 below
   the unadapted 27B judging zero-shot, which the 4B probe beat by far (the 4B
   zero-shot is 0.618). Per source (the best-ID seed, `rolling_w256-soft-s4`,
   against the student-prompt Gleipnir 27B), almost all of the gap is
   SHADE-Arena (0.362 vs 0.588); IaC (0.936 vs 0.987) and CUA-SHADE-Arena
   (0.820 vs 0.881) add a little, and AgentDojo, MLE-Sabotage and Rogue Eval are
   within 0.04 of it.
2. **Backbone effect.** Every family gains 0.06–0.19 OOD over the 4B (the
   selections: soft +0.084, hard +0.105). The useful families also fit faster: ID
   peaked at epochs 1–2 and declined by epoch 8, where the 4B kept improving to
   epochs 8–16.
3. **Soft versus hard.** Soft beats hard by 0.034 Mean-OOD at the selection,
   clearing the 0.01 bar, and in all eight families (0.019–0.055). The gain is
   smaller than at the 4B (+0.055).

| Family, OOD (epoch) | 27B hard | 27B soft | 4B hard | 4B soft |
|---|---:|---:|---:|---:|
| Rolling attention w256 | 0.814 (1) | **0.848** (1) | 0.707 (4) | 0.764 (8) |
| Attention | 0.805 (2) | 0.850 (2) | 0.709 (8) | 0.792 (8) |
| AlphaEvolve | 0.799 (8) | 0.847 (8) | 0.684 (16) | 0.746 (16) |
| Rolling attention w64 | 0.797 (1) | 0.842 (1) | 0.676 (8) | 0.765 (8) |
| MultiMax | 0.793 (4) | 0.842 (8) | 0.599 (8) | 0.684 (16) |
| Rolling attention w10 | 0.773 (1) | 0.820 (1) | 0.636 (2) | 0.689 (8) |
| Mean MLP | 0.566 (8) | 0.621 (8) | 0.406 (16) | 0.458 (16) |
| Mean linear | 0.387 (8) | 0.406 (8) | 0.197 (16) | 0.214 (16) |

Last-token probes behave differently at the 27B. At the 4B they were flat at
about 0.40–0.46 ID across layers. At the 27B, ID climbs from 0.56 at block 31
to a peak of 0.82 at layer 49, then falls back to about 0.78. The selected
layer-49 probes reach 0.517 (hard) and 0.562 (soft) OOD, against 0.21 and 0.20
at the 4B. Block 31 may therefore not be the best layer for the pooled 27B
probes either; this run did not test other layers.

Operations: the first H100 throttled thermally and was replaced (see
[infrastructure](../../docs/infrastructure.md#lambda-cloud)). Extraction on the
replacement took 11.2 hours. The first training attempt failed to pin the
166 GB ID store; the sweep now streams it (`ccca192`, and `e45fc3f` for the
budget extension). Training then took 2.0 hours and the OOD read 29 minutes.
Logs, including the failed attempt (`train_ntp_pinned_oom.log`), are in
`logs/lambda/probe_scale_27b/scale27b_full01/`.

## Prompted follow-up (frozen 2026-10-28, before extraction)

User direction, 2026-10-28: rerun with the full teacher prompt. Almost all of
the NTP probe's OOD gap to Gleipnir 27B is SHADE-Arena, where the probe stays
at 0.36 at both scales. Teacher-prompted Gleipnir 27B gains 14 points there
(0.588 to 0.730), and the unadapted 27B already reaches 0.556 zero-shot. The
question is whether a probe reading the prompted forward taps that judgment.
At the 4B the prompted backbone did not help (selected soft 0.713 vs NTP
0.764, SHADE-Arena about 0.25), but the 4B's zero-shot was weak (0.618).

- **Design.** The NTP run above with only the backbone changed to `prompted`,
  exactly as the 4B `full01` prompted condition: the rows' full teacher
  prompt, chat template, thinking off, ending in `Prediction:`. Probes read the
  trajectory and the suffix at block 31, not the instruction prefix. Same rows,
  probes, seeds, checkpoints, frozen selection rule, budget extension and one
  OOD read. Run `scale27b_prompted01` (`BACKBONE=prompted
  NAME=scale27b_prompted01 run_full.sh`).
- **Parity.** The extraction's zero-shot decision margin should reproduce the
  unadapted 27B's archived teacher-prompt OOD (0.8901, vLLM) within 0.01; a
  larger gap is reported before interpreting the probes. The `parity` split is
  not run: its reference margins are the 4B's.
- **Comparisons.** Prompted vs NTP 27B probes per objective, family and source
  (SHADE-Arena first); prompted 27B vs Gleipnir 27B under the teacher prompt;
  and the 4B prompted-vs-NTP contrast.
- **Store.** Kept on the filesystem for reuse (user decision):
  `stores/qwen38_27b_l31/full/prompted`, about 0.8 TB. The run therefore waits
  for capacity in us-west-3, where the filesystem lives.

## Stop rules

Stop and report in any of these cases:

- extraction fails the block/hidden check or produces nonfinite activations;
- the 27B forward runs out of memory at the longest rows (about 42K tokens);
- local disk runs out (the FP8 store is estimated at 0.87 TB, against 968 GB
  free);
- the persistent copy does not match the local store.

## Operations

`run_full.sh` runs every stage and skips finished ones. The store goes to
`~/stores_27b/full/ntp/` on local disk and is copied to
`/lambda/nfs/gleipnir-probes/stores/qwen38_27b_l31/full/ntp/`. Sweep outputs,
selections and OOD results go to `results/probe_generalization/scale27b_full01/`,
and logs to `logs/lambda/probe_scale_27b/scale27b_full01/`. The run used one
`gpu_1x_h100_pcie` in us-west-3 (`gleipnir-probe-scale27b`, replaced by
`gleipnir-probe-scale27b-2`). Both are terminated; the store stays on the
persistent filesystem.
