# BF16/SDPA B200 training default

User selection, 2026-10-16: train in BF16 with SDPA attention.
`qwen35_4b_b200_default` now aliases `qwen35_4b_b200_packed_sdpa`:
- BF16 MLP and BF16 FlashQLA GDN (FP32 gates/normalization);
- segmented causal SDPA;
- rank 128/alpha 256, 16,384-token packing, logical batch 32, FP32 masters;
- startup reuse of the accepted BF16/SDPA receipt.

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
