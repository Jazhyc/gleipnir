# Probe versus LoRA at 27B: scale pilot

User direction, 2026-10-27: test whether activation probes stay competitive
with LoRA-trained monitors beyond the 4B, starting with a pilot on Qwen3.8-27B.
NTP backbone only; one Lambda GPU with the `gleipnir-probes` filesystem was
authorized. The 4B results this builds on are in
[the probe study](../probe_generalization/README.md).

## Question

At the 4B, the selected NTP soft probe matches the LoRA seeds on ID (0.904 vs
a 0.903 seed mean) and trails them a little on OOD (0.764, against seeds at
0.777–0.813). Does the probe–LoRA gap shrink, hold or grow at 27B, where
Gleipnir 27B reaches 0.9217 ID and 0.9087 OOD? The pilot is descriptive. It
measures the backbone effect at a fixed probe protocol, and it costs the full
27B extraction before anyone commits to it. The user decides on the full run
afterwards.

## Design (frozen 2026-10-27, before any 27B extraction)

- **Backbone.** Unadapted `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
  (64 blocks, hidden 5,120), BF16 Transformers forward, NTP rendering (bare
  trajectory, first token dropped as the attention sink). Residual stream
  after block **31** (zero-based), the same relative depth as the 4B's block 15
  of 32 (user choice). Extraction checks the block count and hidden size at load.
- **Rows** (`prepare.py`, `data/probe_generalization/scale_pilot/`). `train`
  and `id` are byte-identical copies of the probe-study pilot (1,000 rows each,
  hashes checked). `ood` is 1,000 of the 6,395 frozen OOD rows, stratified by
  source and label (seed 0, largest remainder); MLE-Sabotage gets 23 rows.
- **Probes.** The 4B `pilot01` NTP protocol unchanged: the seven pilot
  architectures, hard and Kimi-soft objectives, seeds 0–2, batch 16, AdamW at
  1e-4 / 3e-3, checkpoints at epochs 2, 4, 8 and 16, FP8 storage (accepted for
  the study). Training rows stay in pinned CPU memory and forwards are capped
  at 65,536 tokens with gradient accumulation. That changes memory use only,
  not the updates. No AlphaEvolve probe, matching `pilot01`.
- **4B counterpart.** The existing `pilot01` NTP FP8 probes, unchanged. Only
  the 4B block-15 activations of the OOD sample are newly extracted.
- **Selection and OOD read.** For each backbone and objective, the probe
  study's frozen rule (`evaluate_ood.select`) runs on the 1,000 ID rows. OOD is
  read once (`score.py`) for the selected candidate and for each family's
  ID-best candidate. Per-row logits are kept. Nothing is selected on OOD.

## Comparisons (computed locally, no GPU)

All comparisons use the same 1,000 ID and 1,000 OOD rows, with seed-median
source-macro pAUROC@20:

1. **Backbone effect:** 27B probe minus 4B probe, per objective.
2. **Probe–LoRA gap at each scale:** probe minus Gleipnir 27B (seed 0,
   `qwen38-27b-s0-01`, student prompt) and probe minus the 4B LoRA seeds,
   with a paired bootstrap stratified by source.
3. **Training-data penalty:** the pilot probes train on 1,000 rows, while the
   LoRA monitors use 7,836. At the 4B, `full01` selected probes are scored on
   the same OOD rows (a subset of their already-read logits), which measures
   how much of the pilot gap is data scale.

The non-inferiority margin for a full 27B run (proposed: probe OOD at least
LoRA minus 0.02, with a bootstrap CI) is frozen with that run's contract, not
here.

## Stop rules

Stop and report if extraction fails the block/hidden check or produces
nonfinite activations, if the 27B forward runs out of memory at the longest
rows (about 38K tokens), or if the persistent copy does not match the local
stores.

## Operations

`run_remote.sh` runs every stage in order and skips finished ones. Stores go
to `~/stores_scale/{qwen38_27b,qwen35_4b}/scale_pilot/ntp/` on local disk and
are copied to `/lambda/nfs/gleipnir-probes/stores/scale_pilot/` (about 155 GB
FP8 for the 27B). Sweep outputs go to
`results/probe_generalization/scale27b_pilot01/ntp-fp8/`, selections and OOD
logits to `results/probe_scale_27b/scale27b_pilot01/`, and logs to
`logs/lambda/probe_scale_27b/scale27b_pilot01/`. Instance:
`gleipnir-probe-scale27b`, one `gpu_1x_h100_pcie` in us-west-3.
