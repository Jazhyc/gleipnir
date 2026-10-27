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
and logs to `logs/lambda/probe_scale_27b/scale27b_full01/`. The instance is
`gleipnir-probe-scale27b`, one `gpu_1x_h100_pcie` in us-west-3. The user
authorized terminating it once the run completes and its artifacts are
collected.
