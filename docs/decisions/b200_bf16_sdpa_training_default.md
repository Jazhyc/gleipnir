# BF16/SDPA B200 training default

User selection, 2026-10-16: train in BF16 with SDPA attention.
`qwen35_4b_b200_default` aliases `qwen35_4b_b200_packed_sdpa_trim`:
- BF16 MLP and BF16 FlashQLA GDN (FP32 gates/normalization);
- segmented causal SDPA;
- rank 128/alpha 256, 16,384-token packing, logical batch 32, FP32 masters;
- startup reuse of the accepted BF16/SDPA receipt;
- the compiled final-layer trim (user selection, later on 2026-10-16).

**Final-layer trim.** Layer 31 runs its query, attention output, MLP and
norms only at packed decision positions, while still producing every token's
K/V ([experiment](../../experiments/b200_training_trim/README.md)).
- **Speed.** On the 320-row benchmark, against a same-GPU control (4.924 s),
  updates take 5.3% less time (4.663 s) and peak memory is 3.8 GiB lower.
- **Exactness.** It is not bitwise. Decision logits can differ by one BF16
  ulp, and gradients by about 0.6% on a full pack, against a 0.4% run-to-run
  noise floor.
- **Canary.** Every run checks the trim against the untrimmed layer on a real
  pack and on the longest example before its first update. It stops unless
  gradients agree within that run's noise floor plus 1 point and logits within
  0.125.
- **Isolation gate.** The strict 5% packed/singleton gate on the screen's
  two-example 127/129-token probe fails at 0.055, a one-ulp logit flip. The
  configured 15% tolerance accepts it.
- **Untrimmed recipe.** It stays frozen as `qwen35_4b_b200_packed_sdpa` for
  completed campaigns.

This supersedes the
[2026-10-05 native FP4 default](b200_native_fp4_training_recipe.md). The FP4/FA4
(`qwen35_4b_b200_fp4_mlp`), BF16/FA4 (`qwen35_4b_b200_bf16_fa4`) and FP4/SDPA
(`qwen35_4b_b200_fp4_mlp_sdpa`) profiles remain available for explicit use.
Completed campaigns keep their recorded profiles and receipts.

**Evidence.** Teacher precision factorial on the strict-filtered data
([results](../../experiments/teacher_precision_factorial/README.md)):
- BF16/SDPA (three seeds) has the best ID ranking.
- FA4 lowers OOD macro pAUROC@20 under both MLP precisions (−3.4 / −7.6
  points).
- FP4 lowers ID and the low-noise OOD sources even with SDPA.
- Seed variance is large (OOD macro SD 1.8 points), and the off-diagonal cells
  have one seed.

BF16/SDPA is slower to train than FP4/FA4 (about 50 versus 39 minutes per
epoch on the strict data).
