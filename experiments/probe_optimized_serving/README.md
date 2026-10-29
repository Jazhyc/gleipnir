# Activation probes on the optimized serving stack

User direction, 2026-10-27: measure the frozen 4B NTP probes when the backbone
runs on the optimized B200 serving stack (FP4 MLP/GDN, FP8 attention
projections, MXFP8 attention). Also measure how fast a probe scorer is when
the forward pass stops after the probed block. Quality and throughput are
measured in one session on an EU-RO-1 Runpod B200 (volume `cb7bxu71ug`), which
the user authorized.

## Questions

1. **Quality.** The full01 probes were trained on BF16 Transformers
   activations stored in FP8. Do they keep their ID and OOD pAUROC@20 when
   the activations come from the quantized serving forward? The FP8 gate
   already covered storage quantization. Compute quantization is new: errors
   accumulate through 16 blocks. For reference, Gleipnir 4B v0.3 loses 3.4
   OOD points on this stack (BF16 0.8039, optimized 0.7703), even after QAD.
2. **Throughput.** Under causal attention, blocks 16–31 cannot change block
   15's activations, so stopping after block 15 is exact. How many tokens per
   second does the truncated optimized forward reach on the probe workload,
   compared with the same stack at full depth and with Gleipnir 4B?

## Design (frozen 2026-10-27, before any optimized capture)

- **Backbone.** Unadapted `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`,
  served by the recorded Gleipnir 4B v0.4 optimized command
  (`results/qad-v02-optimized01/server.json`, 32K context) with `--model`
  swapped to the base snapshot. All scales are computed online from BF16
  weights, so no checkpoint-specific calibration changes. Rows use NTP
  rendering, the bare trajectory exactly as in `full01`. Prompt token counts
  must match the Transformers tokenizer row for row.
- **Capture (quality).** The Lens bridge (`gleipnir.serving.lens_worker`,
  eager mode) captures the post-block-15 residual stream at every position,
  before the final norm. As in the reference, the first token (the attention
  sink) is dropped, and activations go through the study's blockwise FP8
  round trip before scoring. Eager mode is a recorded caveat: the Lens
  integration measured eager-versus-compiled score drift of about 0.01–0.04
  on the adapter canary. Block 15 is far from the last-query layer (31), so
  last-query trimming does not touch it.
- **Probes.** The full01 NTP FP8 probe bank (11 architectures × hard/soft ×
  5 seeds) at the checkpoints the frozen picks use (epochs 2, 4, 8 and 16).
  Rows: the 3,012 canonical ID rows and the 6,395 frozen OOD rows
  (`data/probe_generalization/full/`). Reported per pick: the selected
  hard/soft cells (attention e8, rolling_w256 e8) and each family's ID-best
  candidate. The reference is the same probes' BF16-pipeline logits
  (`full01/ntp-fp8/logits_e*_fp8.json` for ID,
  `full01/ensemble_ood/logits_ntp.npz` for OOD). Nothing is selected here.
- **Activation diagnostic.** For 64 stratified OOD rows, the Pod also runs
  the BF16 Transformers extraction (`probe_generalization/extract.py`). This
  gives the per-token relative error and cosine similarity of optimized
  versus BF16 block-15 states.
- **Throughput.** Compiled mode, same command, workload and batching as the
  v0.3 benchmark: the 6,395 OOD rows, sequential 128-row batches at
  concurrency 128, summed batch wall time. Rows are rendered NTP. Measured
  configurations:
  - (a) the base model at full depth;
  - (b) the same model truncated after block 15. A worker subclass sets the
    decoder loop's `end_layer` to 16. All 32 blocks stay loaded, so per-layer
    receipts and counts are unchanged. Its pooled classify output is unused.

  The probe head's own cost (selected soft probe, on GPU) is timed
  separately on captured activations. Gleipnir 4B v0.4's recorded
  226K tok/s on its own prompted workload is the reference. A reported
  speedup states both rows/s and tokens/s, because NTP rows are shorter than
  prompted rows.

Out of scope: training probes on quantized activations (a natural follow-up
if quality drops), the prompted backbone, and the 27B.

## Stop rules

Stop and report in any of these cases:

- token counts differ from the Transformers tokenizer;
- an activation is nonfinite;
- the Lens bridge cannot run on the selected stack;
- truncation changes block-15 activations relative to the full-depth eager
  capture (checked on the diagnostic rows).

## Implementation note (2026-10-27)

Lens works only in eager mode, and eager numerics are not the deployed
compiled numerics, so after an eager pilot the capture moved into the
compiled server. `truncate_worker.py` stops the decoder after block 15 (all
blocks stay loaded), swaps the final norm for an identity that returns the
residual stream, and replaces the classify pooler with the full01 probe
banks, run on the GPU over vLLM's chunk-aware ALL pooling. `start.py` binds
the worker source into the serving compile identity, because vLLM's AOT
cache key ignores module swaps: one truncated launch loaded a full-depth
graph and failed (`thr01/server_trunc.failed_aot_cache_collision.log`).
`probe_client.py` sends rows to `/classify` and saves logits and timing;
`compare.py` scores the frozen picks (run it on the Pod, not the VPS).

## Result (2026-10-27, `opt01`, `thr01`, `diag01`)

**Validation.** The Pod's BF16 extraction, scored offline, reproduces the
full01 reference logits (correlation 0.9998). Compiled and eager optimized
logits agree at 0.980, and both differ from BF16 by about 0.04 in correlation,
so the shift is quantization, not plumbing. On the 64 diagnostic rows,
block-15 states from the optimized stack have mean relative L2 error 0.47 and
mean cosine 0.86 against BF16; the FP8 storage round trip is 0.022 / 0.9998.
The error is about 3% at the first tokens and does not grow with position.

**Quality** (all ID and OOD rows, frozen picks, seed-median pAUROC@20,
paired source-stratified bootstrap):

| cell | ID BF16 → opt | OOD BF16 → opt | OOD change (95% CI) |
|---|---|---|---|
| hard attention e8 (selected) | 0.853 → 0.802 | 0.709 → 0.659 | −0.051 [−0.066, −0.039] |
| soft rolling_w256 e8 (selected) | 0.904 → 0.852 | 0.764 → 0.711 | −0.054 [−0.067, −0.043] |
| soft attention e8 (family) | 0.881 → 0.835 | 0.792 → 0.758 | −0.034 |

Every family loses ID and OOD (OOD family drops −0.008 to −0.076; mean-pooled
probes least). Raw versus FP8-round-tripped inputs differ by at most 0.0004.
The loss matches Gleipnir 4B v0.2 on the same stack before QAD (OOD 0.811 →
0.758), so probes are not more robust to quantized compute than the LoRA.
Details: `opt01/comparison_{fp8,raw}.json`.

**Throughput** (one B200, 6,395 OOD rows, 128-row batches at concurrency 128):

| configuration | tok/s | rows/s |
|---|---:|---:|
| full depth, prompted workload | 228.4K | 26.4 |
| full depth, NTP rows | 224.7K | 26.4 |
| truncated after block 15, no probe | 404.4K | 47.6 |
| truncated + 110-probe bank, FP8 trip | 318.9K | 37.5 |
| truncated + deployed probe (soft attention s1, raw) | 412.5K | 48.5 |

The full-depth prompted pass reproduces Gleipnir 4B v0.3/v0.4's recorded
226K tok/s. The deployed single probe (user choice: soft attention, its
family's ID-best checkpoint and best-ID seed) costs nothing measurable over
the truncated backbone, giving 1.84× the rows per second of the full stack.
Its logits match the same probe in the quality pass at correlation 0.996, not
exactly, because the stack's FP4 arithmetic depends on how many rows share a
step; its OOD pAUROC@20 here is 0.750.
